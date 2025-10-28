from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate
from django.contrib.auth.models import User
from django.http import JsonResponse, HttpResponse, HttpResponseForbidden
from django.db.models import Count, Q, Min, Max
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework.response import Response
from rest_framework import status
from datetime import date, datetime, time, timedelta
import io
import csv
import json 
import os 
import uuid 

# Thêm import cho deepface và numpy
try:
    print("Attempting to import DeepFace and NumPy...")
    from deepface import DeepFace
    import numpy as np
    from numpy.linalg import norm as l2_norm # Import L2 norm
    print("DeepFace and NumPy imported successfully.")
except Exception as e:
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("CRITICAL ERROR: Failed to import 'deepface' or 'numpy'.")
    print(f"The actual error is: {e}")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    DeepFace = None
    np = None
    l2_norm = None

# Thêm import để xử lý file tạm
from django.core.files.storage import default_storage

from .models import Department, Position, Role, WorkLocation, Shift, Employee, Attendance, AttendanceChangeLog
from .serializers import (
    EmployeeMeSerializer, EmployeeSerializer, AttendanceSerializer, WorkLocationSerializer, ShiftSerializer
)
from .utils import haversine_m, week_bounds, month_bounds

# Cấu hình model deepface
FACE_MODEL_NAME = "VGG-Face"
FACE_DISTANCE_THRESHOLD = 0.40 # Ngưỡng cho FaceNet (Cosine Distance)

# ----------------- HÀM HELPER -----------------

def _findCosineDistance(source_representation, test_representation):
    """
    Tính toán Cosine Distance bằng NumPy.
    """
    if np is None or l2_norm is None:
        raise ImportError("NumPy library not loaded correctly.")
        
    a = np.asarray(source_representation)
    b = np.asarray(test_representation)
    dot_product = np.dot(a, b)
    norm_a = l2_norm(a)
    norm_b = l2_norm(b)
    similarity = dot_product / (norm_a * norm_b)
    distance = 1.0 - similarity
    return distance

def _enroll_face_helper(employee_instance, image_file):
    """
    Hàm trợ giúp xử lý file ảnh, tạo embedding và lưu vào employee.
    """
    if DeepFace is None:
        return (False, "Lỗi: Thư viện 'deepface' chưa được cài đặt trên server.")

    tmp_name = f"tmp/{uuid.uuid4()}_{image_file.name}"
    tmp_path = default_storage.save(tmp_name, image_file)
    tmp_full_path = default_storage.path(tmp_path)
    
    error_message = None
    success = False

    try:
        # ---- THAY ĐỔI: Dùng detector_backend='mtcnn' và enforce_detection=False ----
        results = DeepFace.represent(
            img_path=tmp_full_path, 
            model_name=FACE_MODEL_NAME, 
            enforce_detection=False, # Sửa: Tắt báo lỗi, để chúng ta tự kiểm tra
            detector_backend='mtcnn'  # Sửa: Chỉ định rõ detector là 'mtcnn'
        )
        
        # 'results' bây giờ là một list.
        # Chúng ta cần kiểm tra xem nó có rỗng (không tìm thấy) hay nhiều hơn 1
        if not results:
            error_message = "Không nhận diện được khuôn mặt. Vui lòng thử ảnh khác rõ nét và chính diện hơn."
        elif len(results) > 1:
            error_message = "Phát hiện có quá nhiều khuôn mặt. Vui lòng thử ảnh chỉ có một người."
        else:
            # Thành công, results có 1 phần tử
            embedding = results[0]['embedding']
            employee_instance.face_embedding = json.dumps(embedding)
            employee_instance.save() 
            success = True
        
    except Exception as e:
        # Bắt các lỗi khác (ví dụ: file ảnh bị hỏng)
        error_message = f"Lỗi xử lý ảnh: {str(e)}"
    finally:
        if default_storage.exists(tmp_path):
            default_storage.delete(tmp_path)
    
    return (success, error_message)

def user_has_role(user, *roles):
    # (Hàm này giữ nguyên)
    if user.is_superuser:
        return True
    try:
        emp = user.employee
    except Exception:
        return False
    if emp.role and emp.role.name in roles:
        return True
    return False

def require_roles(*roles):
    # (Hàm này giữ nguyên)
    def _decorator(view_func):
        def _wrapped(request, *args, **kwargs):
            if not request.user.is_authenticated:
                from django.contrib.auth.views import redirect_to_login
                return redirect_to_login(request.get_full_path())
            if not user_has_role(request.user, *roles):
                return HttpResponseForbidden("Bạn không có quyền truy cập chức năng này.")
            return view_func(request, *args, **kwargs)
        return _wrapped
    return _decorator

# ---------------- API (Cập nhật api_clock) -----------------
@api_view(["POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def api_clock(request):
    if DeepFace is None or np is None:
        return Response({"ok": False, "message": "Tính năng nhận diện khuôn mặt chưa được cài đặt trên server (DeepFace/NumPy)."}, status=500)

    emp = get_object_or_404(Employee, user=request.user, is_active=True)
    
    # --- 1. XÁC THỰC KHUÔN MẶT ---
    if not emp.face_embedding:
        return Response({"ok": False, "message": "Tài khoản của bạn chưa đăng ký khuôn mặt. Vui lòng liên hệ quản trị."}, status=400)
    
    if 'face_image' not in request.FILES:
        return Response({"ok": False, "message": "Yêu cầu hình ảnh khuôn mặt để chấm công."}, status=400)

    live_image_file = request.FILES['face_image']
    
    tmp_live_path = default_storage.save(f"tmp/{live_image_file.name}", live_image_file)
    tmp_live_full_path = default_storage.path(tmp_live_path)

    try:
        # Load mẫu đã lưu từ DB
        stored_embedding = json.loads(emp.face_embedding)
        
        # ---- THAY ĐỔI: Dùng detector_backend='mtcnn' và enforce_detection=False ----
        live_results = DeepFace.represent(
            img_path=tmp_live_full_path, 
            model_name=FACE_MODEL_NAME, 
            enforce_detection=False, # Sửa: Tắt báo lỗi
            detector_backend='mtcnn'  # Sửa: Chỉ định 'mtcnn'
        )
        
        if not live_results:
            return Response({"ok": False, "message": "Không nhận diện được khuôn mặt trong ảnh bạn gửi."}, status=400)
        if len(live_results) > 1:
            return Response({"ok": False, "message": "Phát hiện nhiều khuôn mặt trong ảnh chấm công."}, status=400)
        
        # Thành công, lấy embedding
        live_embedding = live_results[0]['embedding']
        
        # Sử dụng hàm helper _findCosineDistance (bằng NumPy)
        distance = _findCosineDistance(stored_embedding, live_embedding)

        if distance > FACE_DISTANCE_THRESHOLD:
             return Response({"ok": False, "message": f"Xác thực khuôn mặt thất bại (Khoảng cách: {distance:.2f}). Đây không phải bạn."}, status=400)
        
    except Exception as e:
        # Bắt các lỗi khác (file hỏng,...)
        return Response({"ok": False, "message": f"Lỗi xử lý ảnh: {str(e)}"}, status=500)
    finally:
        if default_storage.exists(tmp_live_path):
            default_storage.delete(tmp_live_path)

    # --- 2. XÁC THỰC VỊ TRÍ (Logic cũ, giữ nguyên) ---
    lat = float(request.POST.get("latitude"))
    lon = float(request.POST.get("longitude"))
    t = request.POST.get("type", None) 
    work_location_id = request.POST.get("work_location_id", None)
    
    if work_location_id is None:
        loc = emp.allowed_locations.first()
        if not loc:
            return Response({"ok": False, "message": "Bạn chưa được cấu hình địa điểm chấm công."}, status=400)
    else:
        loc = get_object_or_404(WorkLocation, pk=int(work_location_id))

    if not emp.allowed_locations.filter(pk=loc.pk).exists():
        return Response({"ok": False, "message": "Địa điểm này không thuộc phạm vi được phép."}, status=400)

    distance = haversine_m(lat, lon, loc.latitude, loc.longitude)
    within = distance <= loc.radius_m

    if t not in ["IN","OUT"]:
        today = timezone.localdate()
        last_in = Attendance.objects.filter(employee=emp, type="IN", timestamp__date=today).order_by("-timestamp").first()
        last_out = Attendance.objects.filter(employee=emp, type="OUT", timestamp__date=today).order_by("-timestamp").first()
        t = "OUT" if last_in and (not last_out or last_in.timestamp > last_out.timestamp) else "IN"

    att = Attendance.objects.create(
        employee=emp, type=t, latitude=lat, longitude=lon,
        distance_m=round(distance,2), within_geofence=within, work_location=loc, created_by=request.user
    )

    return Response({
        "ok": True, "within_geofence": within, "distance_m": round(distance,2), "type": t,
        "timestamp": att.timestamp, "work_location": WorkLocationSerializer(loc).data
    })


# ... (Tất cả các hàm còn lại: api_employee_me, api_change_password, api_history, ...)
# ... (web_dashboard, web_monitor, web_employees, web_employee_edit, ...)
# ... (đều giữ nguyên như phiên bản trước mà tôi đã cung cấp)

@api_view(["GET","PATCH"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def api_employee_me(request):
    emp, _ = Employee.objects.get_or_create(user=request.user, defaults={"is_active": True})
    if request.method == "GET":
        # ---- THÊM DÒNG PRINT 1 ----
        print(f"DEBUG: Employee {emp.user.username} allowed locations BEFORE serialization: {list(emp.allowed_locations.all())}")
        
        serializer = EmployeeMeSerializer(emp)
        
        # ---- THÊM DÒNG PRINT 2 ----
        print(f"DEBUG: Serializer data for allowed_locations: {serializer.data.get('allowed_locations')}") 
        
        return Response(serializer.data)
        
    # PATCH logic (giữ nguyên)
    user = request.user
    user.first_name = request.data.get("first_name", user.first_name)
    user.last_name = request.data.get("last_name", user.last_name)
    user.email = request.data.get("email", user.email)
    user.save()
    emp.phone = request.data.get("phone", emp.phone)
    emp.save()
    return Response(EmployeeMeSerializer(emp).data)

@api_view(["PATCH"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def api_employee_me_patch(request):
    pass

@api_view(["PATCH", "POST"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def api_change_password(request):
    # (Giữ nguyên)
    new1 = request.data.get("new_password1")
    new2 = request.data.get("new_password2")
    if not new1 or new1 != new2:
        return Response({"ok": False, "message": "Mật khẩu nhập lại không khớp."}, status=400)
    request.user.set_password(new1)
    request.user.save()
    return Response({"ok": True})

@api_view(["GET"])
@authentication_classes([JWTAuthentication])
@permission_classes([IsAuthenticated])
def api_history(request):
    # (Giữ nguyên)
    emp = get_object_or_404(Employee, user=request.user)
    period = request.GET.get("period", "day")
    date_str = request.GET.get("date")
    if date_str:
        try:
            base_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except:
            base_date = timezone.localdate()
    else:
        base_date = timezone.localdate()

    if period == "week":
        start, end = week_bounds(base_date)
    elif period == "month":
        start, end = month_bounds(base_date)
    else:
        start = base_date
        end = base_date

    qs = Attendance.objects.filter(employee=emp, timestamp__date__gte=start, timestamp__date__lte=end).order_by("timestamp")
    days = {}
    for a in qs:
        d = a.timestamp.date()
        days.setdefault(d, []).append(a)

    results = []
    total_hours_all = 0.0
    
    for d, items in sorted(days.items()):
        total_hours = 0.0
        ins = [x for x in items if x.type=="IN"]
        outs = [x for x in items if x.type=="OUT"]
        i = 0; j = 0
        while i < len(ins) and j < len(outs):
            if ins[i].timestamp <= outs[j].timestamp:
                duration = (outs[j].timestamp - ins[i].timestamp).total_seconds()/3600.0
                total_hours += max(0.0, duration)
                i += 1; j += 1
            else:
                j += 1
        total_hours_all += total_hours

        late = False
        early_leave = False
        shift = emp.shift
        if shift:
            st = timezone.make_aware(datetime.combine(d, shift.start_time))
            en = timezone.make_aware(datetime.combine(d, shift.end_time))
            grace_in = timedelta(minutes=shift.late_grace_min)
            grace_out = timedelta(minutes=shift.early_grace_min)
            first_in = ins[0].timestamp if ins else None
            last_out = outs[-1].timestamp if outs else None
            if first_in and first_in > st + grace_in:
                late = True
            if last_out and last_out < en - grace_out:
                early_leave = True

        results.append({
            "date": d, 
            "items": AttendanceSerializer(items, many=True).data,
            "total_hours": round(total_hours, 2),
            "late": late,
            "early_leave": early_leave,
        })

    return Response({
        "period": period,
        "start": start, "end": end,
        "days": results,
        "sum_hours": round(total_hours_all,2)
    })

# ---------------- Web UI (Giữ nguyên) -----------------
@login_required
def web_dashboard(request):
    # (Giữ nguyên)
    date_str = request.GET.get("date")
    view = request.GET.get("view", "day")
    if date_str:
        base = datetime.strptime(date_str, "%Y-%m-%d").date()
    else:
        base = timezone.localdate()

    if view == "month":
        start, end = month_bounds(base)
    elif view == "year":
        start = base.replace(month=1, day=1)
        end = base.replace(month=12, day=31)
    else:
        start, end = base, base

    total_emp = Employee.objects.filter(is_active=True).count()
    present_ids = set(Attendance.objects.filter(type="IN", timestamp__date__gte=start, timestamp__date__lte=end).values_list("employee_id", flat=True))
    present = len(present_ids)
    absent = max(0, total_emp - present)
    late_count = 0
    for emp in Employee.objects.filter(is_active=True, shift__isnull=False):
        ins = Attendance.objects.filter(employee=emp, type="IN", timestamp__date__gte=start, timestamp__date__lte=end).order_by("timestamp")
        if ins.exists():
            d = ins.first().timestamp.date()
            st = timezone.make_aware(datetime.combine(d, emp.shift.start_time))
            if ins.first().timestamp > st + timedelta(minutes=emp.shift.late_grace_min):
                late_count += 1

    daily_hours = []
    days = Attendance.objects.filter(timestamp__date__gte=start, timestamp__date__lte=end).dates("timestamp", "day")
    for d in days:
        items = Attendance.objects.filter(timestamp__date=d).order_by("timestamp")
        hours = 0.0
        emps = Employee.objects.all()
        for emp in emps:
            emp_items = items.filter(employee=emp)
            ins = list(emp_items.filter(type="IN"))
            outs = list(emp_items.filter(type="OUT"))
            i = 0; j = 0
            emp_hours = 0.0
            while i < len(ins) and j < len(outs):
                if ins[i].timestamp <= outs[j].timestamp:
                    emp_hours += max(0.0, (outs[j].timestamp - ins[i].timestamp).total_seconds()/3600.0)
                    i += 1; j += 1
                else:
                    j += 1
            hours += emp_hours
        daily_hours.append((d, hours))

    context = {
        "total_emp": total_emp,
        "present": present,
        "absent": absent,
        "late_count": late_count,
        "date": base,
        "view": view,
        "daily_hours": daily_hours,
    }
    return render(request, "attendance/dashboard.html", context)

@login_required
def web_monitor(request):
    # (Giữ nguyên)
    recs = Attendance.objects.select_related("employee","work_location").order_by("-timestamp")[:100]
    return render(request, "attendance/monitor.html", {"records": recs})

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_employees(request):
    # (Giữ nguyên từ lần sửa trước)
    if request.method == "POST" and "action" in request.POST and request.POST["action"] == "create":
        username = request.POST["username"]
        first_name = request.POST.get("first_name","")
        last_name = request.POST.get("last_name","")
        email = request.POST.get("email","")
        phone = request.POST.get("phone","")
        role_id = request.POST.get("role_id")
        shift_id = request.POST.get("shift_id")
        dept_id = request.POST.get("department_id")
        pos_id = request.POST.get("position_id")

        if User.objects.filter(username=username).exists():
            employees = Employee.objects.select_related("user","role","shift","department","position").all().order_by("user__username")
            roles = Role.objects.all()
            shifts = Shift.objects.all()
            locations = WorkLocation.objects.all()
            departments = Department.objects.all()
            positions = Position.objects.all()
            return render(request, "attendance/employees.html", {"error":"Username đã tồn tại.", "employees": employees, "roles": roles, "shifts": shifts, "locations": locations, "departments": departments, "positions": positions} )

        user = User.objects.create_user(username=username, password="12345678", first_name=first_name, last_name=last_name, email=email)
        emp = Employee.objects.create(user=user, phone=phone, is_active=True,
                                      role_id=role_id if role_id else None, shift_id=shift_id if shift_id else None,
                                      department_id=dept_id if dept_id else None, position_id=pos_id if pos_id else None)
        loc_ids = request.POST.getlist("allowed_location_ids")
        if loc_ids:
            emp.allowed_locations.set(WorkLocation.objects.filter(id__in=loc_ids))
        emp.save()

        if 'face_image' in request.FILES and request.FILES['face_image']:
            image_file = request.FILES['face_image']
            success, error = _enroll_face_helper(emp, image_file)
            
            if not success:
                employees = Employee.objects.select_related("user","role","shift","department","position").all().order_by("user__username")
                roles = Role.objects.all()
                shifts = Shift.objects.all()
                locations = WorkLocation.objects.all()
                departments = Department.objects.all()
                positions = Position.objects.all()
                context = {
                    "employees": employees, "roles": roles, "shifts": shifts, "locations": locations, "departments": departments, "positions": positions,
                    "error": f"Tạo người dùng {username} thành công, NHƯNG đăng ký khuôn mặt thất bại: {error}"
                }
                return render(request, "attendance/employees.html", context)
        
        return redirect("web_employees")

    employees = Employee.objects.select_related("user","role","shift","department","position").all().order_by("user__username")
    roles = Role.objects.all()
    shifts = Shift.objects.all()
    locations = WorkLocation.objects.all()
    departments = Department.objects.all()
    positions = Position.objects.all()
    return render(request, "attendance/employees.html", {
        "employees": employees, "roles": roles, "shifts": shifts, "locations": locations, "departments": departments, "positions": positions
    })

@login_required
def web_employee_new(request):
    # (Giữ nguyên)
    return redirect("web_employees")

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_employee_edit(request, pk):
    # (Giữ nguyên)
    emp = get_object_or_404(Employee, pk=pk)
    if request.method == "POST":
        emp.phone = request.POST.get("phone", emp.phone)
        emp.is_active = bool(request.POST.get("is_active", "1") == "1")
        emp.role_id = request.POST.get("role_id") or None
        emp.shift_id = request.POST.get("shift_id") or None
        emp.department_id = request.POST.get("department_id") or None
        emp.position_id = request.POST.get("position_id") or None
        loc_ids = request.POST.getlist("allowed_location_ids")
        emp.allowed_locations.set(WorkLocation.objects.filter(id__in=loc_ids))
        emp.save()
        user = emp.user
        user.first_name = request.POST.get("first_name", user.first_name)
        user.last_name = request.POST.get("last_name", user.last_name)
        user.email = request.POST.get("email", user.email)
        user.save()
        return redirect("web_employees")
    roles = Role.objects.all()
    shifts = Shift.objects.all()
    locations = WorkLocation.objects.all()
    departments = Department.objects.all()
    positions = Position.objects.all()
    return render(request, "attendance/employee_edit.html", {"emp": emp, "roles": roles, "shifts": shifts, "locations": locations, "departments": departments, "positions": positions})

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_employee_enroll_face(request, pk):
    # (Giữ nguyên từ lần sửa trước)
    if DeepFace is None:
        return HttpResponse("Lỗi: Thư viện 'deepface' chưa được cài đặt trên server.", status=500)
        
    emp = get_object_or_404(Employee, pk=pk)
    context = {"emp": emp, "error": None}
    
    if request.method == "POST":
        if 'face_image' not in request.FILES or not request.FILES['face_image']:
            context["error"] = "Bạn chưa chọn file ảnh."
            return render(request, "attendance/web_employee_enroll_face.html", context)
        
        image_file = request.FILES['face_image']
        success, error = _enroll_face_helper(emp, image_file)
            
        if success:
            return redirect("web_employee_edit", pk=emp.id)
        else:
            context["error"] = error

    return render(request, "attendance/web_employee_enroll_face.html", context)


@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_employee_toggle(request, pk):
    # (Giữ nguyên)
    emp = get_object_or_404(Employee, pk=pk)
    emp.is_active = not emp.is_active
    emp.save()
    return redirect("web_employees")

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_employee_reset_password(request, pk):
    # (Giữ nguyên)
    emp = get_object_or_404(Employee, pk=pk)
    emp.user.set_password("12345678")
    emp.user.save()
    return redirect("web_employees")

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_shifts(request):
    # (Giữ nguyên)
    if request.method == "POST":
        sid = request.POST.get("id")
        data = {
            "name": request.POST.get("name"),
            "start_time": request.POST.get("start_time"),
            "end_time": request.POST.get("end_time"),
            "break_minutes": int(request.POST.get("break_minutes") or 0),
            "late_grace_min": int(request.POST.get("late_grace_min") or 5),
            "early_grace_min": int(request.POST.get("early_grace_min") or 5),
        }
        if sid:
            s = get_object_or_404(Shift, pk=sid)
            for k,v in data.items():
                setattr(s, k, v)
            s.save()
        else:
            Shift.objects.create(**data)
        return redirect("web_shifts")
    shifts = Shift.objects.all().order_by("name")
    return render(request, "attendance/shifts.html", {"shifts": shifts})

@login_required
@require_roles('Quản trị viên','Nhân sự')
def web_locations(request):
    # (Giữ nguyên)
    if request.method == "POST":
        lid = request.POST.get("id")
        data = {
            "name": request.POST.get("name"),
            "latitude": float(request.POST.get("latitude")),
            "longitude": float(request.POST.get("longitude")),
            "radius_m": int(request.POST.get("radius_m")),
        }
        if lid:
            loc = get_object_or_404(WorkLocation, pk=lid)
            for k,v in data.items():
                setattr(loc, k, v)
            loc.save()
        else:
            WorkLocation.objects.create(**data)
        return redirect("web_locations")
    locations = WorkLocation.objects.all().order_by("name")
    return render(request, "attendance/locations.html", {"locations": locations})

@login_required
@require_roles('Quản trị viên','Nhân sự','Trưởng phòng')
def web_attendance_edit(request, pk):
    # (Giữ nguyên)
    a = get_object_or_404(Attendance, pk=pk)
    if request.method == "POST":
        before = AttendanceSerializer(a).data
        a.type = request.POST.get("type", a.type)
        a.timestamp = timezone.make_aware(datetime.strptime(request.POST.get("timestamp"), "%Y-%m-%d %H:%M"))
        a.latitude = float(request.POST.get("latitude"))
        a.longitude = float(request.POST.get("longitude"))
        a.work_location_id = int(request.POST.get("work_location_id"))
        a.note = request.POST.get("note","")
        a.changed_by = request.user
        a.changed_at = timezone.now()
        a.save()
        AttendanceChangeLog.objects.create(attendance=a, action="edited", reason=request.POST.get("reason",""), before_data=before, after_data=AttendanceSerializer(a).data, changed_by=request.user)
        return redirect("web_monitor")
    locations = WorkLocation.objects.all()
    return render(request, "attendance/attendance_edit.html", {"a": a, "locations": locations})

@login_required
@require_roles('Quản trị viên','Nhân sự','Trưởng phòng')
def web_attendance_new(request):
    # (Giữ nguyên)
    if request.method == "POST":
        emp_id = int(request.POST.get("employee_id"))
        emp = get_object_or_404(Employee, pk=emp_id)
        a = Attendance.objects.create(
            employee=emp,
            type=request.POST.get("type","IN"),
            timestamp=timezone.make_aware(datetime.strptime(request.POST.get("timestamp"), "%Y-%m-%d %H:%M")),
            latitude=float(request.POST.get("latitude")),
            longitude=float(request.POST.get("longitude")),
            work_location_id=int(request.POST.get("work_location_id")),
            note=request.POST.get("note",""),
            created_by=request.user
        )
        AttendanceChangeLog.objects.create(attendance=a, action="created", reason=request.POST.get("reason",""), after_data=AttendanceSerializer(a).data, changed_by=request.user)
        return redirect("web_monitor")
    employees = Employee.objects.select_related("user").all()
    locations = WorkLocation.objects.all()
    return render(request, "attendance/attendance_new.html", {"employees": employees, "locations": locations})

@login_required
@require_roles('Quản trị viên','Nhân sự','Trưởng phòng')
def web_monthly(request):
    # (Giữ nguyên)
    month = request.GET.get("month")
    if month:
        y, m = [int(x) for x in month.split("-")]
        d = date(y, m, 1)
    else:
        d = timezone.localdate().replace(day=1)
    start, end = d.replace(day=1), month_bounds(d)[1]
    employees = Employee.objects.filter(is_active=True).select_related("user","shift")
    days = [(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    table = []
    for emp in employees:
        row = {"employee": emp, "daily": [], "total": 0.0}
        for day in days:
            items = Attendance.objects.filter(employee=emp, timestamp__date=day).order_by("timestamp")
            ins = list(items.filter(type="IN"))
            outs = list(items.filter(type="OUT"))
            i=j=0; hours=0.0
            while i < len(ins) and j < len(outs):
                if ins[i].timestamp <= outs[j].timestamp:
                    hours += (outs[j].timestamp - ins[i].timestamp).total_seconds()/3600.0
                    i+=1; j+=1
                else:
                    j+=1
            row["daily"].append(round(hours,2))
            row["total"] += hours
        row["total"] = round(row["total"],2)
        table.append(row)
    return render(request, "attendance/monthly.html", {"days": days, "table": table, "month": d.strftime("%Y-%m")})

@login_required
@require_roles('Quản trị viên','Nhân sự','Trưởng phòng')
def web_monthly_export(request):
    # (Giữ nguyên)
    month = request.GET.get("month")
    if month:
        y, m = [int(x) for x in month.split("-")]
        d = date(y, m, 1)
    else:
        d = timezone.localdate().replace(day=1)
    start, end = d.replace(day=1), month_bounds(d)[1]
    employees = Employee.objects.filter(is_active=True).select_related("user","shift")

    days = [(start + timedelta(days=i)) for i in range((end - start).days + 1)]
    output = io.StringIO()
    writer = csv.writer(output)
    header = ["Username", "Họ tên"] + [x.strftime("%d/%m") for x in days] + ["Tổng giờ"]
    writer.writerow(header)

    for emp in employees:
        row = [emp.user.username, emp.user.get_full_name()]
        total = 0.0
        for day in days:
            items = Attendance.objects.filter(employee=emp, timestamp__date=day).order_by("timestamp")
            ins = list(items.filter(type="IN"))
            outs = list(items.filter(type="OUT"))
            i=j=0; hours=0.0
            while i < len(ins) and j < len(outs):
                if ins[i].timestamp <= outs[j].timestamp:
                    hours += (outs[j].timestamp - ins[i].timestamp).total_seconds()/3600.0
                    i+=1; j+=1
                else:
                    j+=1
            row.append(round(hours,2))
            total += hours
        row.append(round(total,2))
        writer.writerow(row)

    resp = HttpResponse(output.getvalue(), content_type="text/csv")
    resp['Content-Disposition'] = f'attachment; filename="bang_cong_{d:%Y_%m}.csv"'
    return resp

from django.contrib.auth import login as auth_login, logout as auth_logout

def web_login(request):
    # (Giữ nguyên)
    if request.method == "POST":
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)
        if user:
            auth_login(request, user)
            return redirect(request.GET.get("next") or "/web/dashboard/")
        else:
            return render(request, "attendance/login.html", {"error":"Sai tài khoản hoặc mật khẩu."})
    return render(request, "attendance/login.html")

def web_logout(request):
    # (Giữ nguyên)
    auth_logout(request)
    return redirect("/web/login/")