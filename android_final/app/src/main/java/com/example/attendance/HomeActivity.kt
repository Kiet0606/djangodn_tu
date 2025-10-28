
package com.example.attendance

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.location.Location
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.FileProvider
import android.util.Log
import com.google.android.gms.location.LocationServices
import okhttp3.MediaType.Companion.toMediaTypeOrNull // Thêm
import okhttp3.MultipartBody // Thêm
import okhttp3.RequestBody.Companion.toRequestBody // Thêm
import retrofit2.Call
import retrofit2.Callback
import retrofit2.Response
import java.io.File

class HomeActivity : AppCompatActivity() {

    private var selectedWorkLocationId: Int? = null
    private var allowedLocations: List<WorkLocation> = emptyList()


    private lateinit var statusText: TextView
    private val fused by lazy { LocationServices.getFusedLocationProviderClient(this) } // Thêm


    private val requestPermission = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { perms ->
        if (perms[Manifest.permission.ACCESS_FINE_LOCATION] == true ||
            perms[Manifest.permission.ACCESS_COARSE_LOCATION] == true) {
            requestCameraAndClock()
        } else {
            Toast.makeText(this, "Cần quyền vị trí để chấm công", Toast.LENGTH_SHORT).show()
        }
    }

    // 2. Contract xin quyền (camera) - MỚI
    private val requestCameraPermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            // Nếu quyền camera OK, mở camera
            launchCamera()
        } else {
            Toast.makeText(this, "Cần quyền camera để chấm công", Toast.LENGTH_SHORT).show()
        }
    }

    // 3. Contract chụp ảnh - MỚI
    private var tempImageUri: Uri? = null

    private val takePicture = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success) {
            tempImageUri?.let { uri ->
                // Chụp ảnh thành công -> gọi hàm clock
                clock(uri)
            }
        } else {
            statusText.text = "Đã hủy chụp ảnh."
            Toast.makeText(this, "Hủy chụp ảnh", Toast.LENGTH_SHORT).show()
        }
    }

    // Hàm helper tạo file tạm - MỚI
    // Hàm helper tạo file tạm - MỚI
    private fun createTempImageUri(): Uri? { // Sửa: Trả về Uri? (nullable)
        return try { // ---- THÊM TRY-CATCH ----
            // ---- THAY ĐỔI: Dùng cacheDir thay vì externalCacheDir ----
            val file = File(cacheDir, "temp_face_${System.currentTimeMillis()}.jpg")
            val authority = "${applicationContext.packageName}.provider"
            FileProvider.getUriForFile(this, authority, file)
        } catch (e: Exception) {
            Log.e("HomeActivity", "Error creating temp image URI", e) // Log lỗi
            Toast.makeText(this, "Lỗi tạo file ảnh tạm: ${e.message}", Toast.LENGTH_LONG).show()
            null // Trả về null nếu có lỗi
        }
    }

    // --- KẾT THÚC CONTRACTS ---

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_home)

        statusText = findViewById(R.id.statusText)
        findViewById<Button>(R.id.historyBtn).setOnClickListener { startActivity(Intent(this, HistoryActivity::class.java)) }
        findViewById<Button>(R.id.profileBtn).setOnClickListener { startActivity(Intent(this, ProfileActivity::class.java)) }
        findViewById<Button>(R.id.clockBtn).setOnClickListener { requestLocationAndClock() }

        supportActionBar?.setDisplayHomeAsUpEnabled(true)

        // Fetch user info to greet
        val api = RetrofitClient.retrofit(this).create(ApiService::class.java)
        api.me().enqueue(object : Callback<EmployeeMe> {
            override fun onResponse(call: Call<EmployeeMe>, response: Response<EmployeeMe>) {
                if (response.isSuccessful) {
                    val me = response.body() // Lấy body, có thể null
                    if (me != null) { // ---- THÊM KIỂM TRA NULL ----
                        findViewById<TextView>(R.id.welcomeText).text = "Xin chào, ${me.username}"

                        // ---- THÊM LOG ĐỂ KIỂM TRA ----
                        Log.d("HomeActivity", "Allowed locations received: ${me.allowed_locations}")

                        // Gán giá trị, kiểm tra null trước khi gán để tránh lỗi nếu JSON lỗi
                        allowedLocations = me.allowed_locations ?: emptyList()

                        if (allowedLocations.isEmpty()) {
                            statusText.text = "Bạn chưa được cấu hình địa điểm chấm công. Liên hệ quản trị để gán địa điểm."
                            Log.w("HomeActivity", "allowedLocations list is empty after API call.") // Thêm log cảnh báo
                        } else {
                            // Chọn mặc định địa điểm đầu tiên
                            selectedWorkLocationId = allowedLocations.first().id
                            statusText.text = "Sẵn sàng chấm công tại: ${allowedLocations.first().name}" // Cập nhật status
                            Log.i("HomeActivity", "Default location set to: ${allowedLocations.first().name} (ID: $selectedWorkLocationId)") // Thêm log thành công
                        }
                    } else {
                        // Trường hợp body null dù isSuccessful (rất hiếm)
                        statusText.text = "Không lấy được thông tin người dùng (body null)"
                        Log.e("HomeActivity", "API call successful but response body is null.")
                    }
                } else {
                    statusText.text = "Không lấy được thông tin người dùng (${response.code()})"
                    Log.e("HomeActivity", "API call failed with code: ${response.code()}")
                }
            }
            override fun onFailure(call: Call<EmployeeMe>, t: Throwable) {
                statusText.text = "Lỗi lấy thông tin: ${t.message}"
                Log.e("HomeActivity", "API call failed with error: ${t.message}", t) // Thêm log lỗi chi tiết
            }
        })
    }

    private fun requestLocationAndClock() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            requestPermission.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION))
            return
        }
        requestCameraAndClock()
    }

    private fun requestCameraAndClock() {
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            // Xin quyền camera
            requestCameraPermission.launch(Manifest.permission.CAMERA)
            return
        }
        // Nếu đã có cả 2 quyền, mở camera
        launchCamera()
    }

    // HÀM MỚI (3): Mở camera
    private fun launchCamera() {
        statusText.text = "Đang mở camera..."
        tempImageUri = createTempImageUri() // Có thể trả về null

        // ---- THÊM KIỂM TRA NULL ----
        if (tempImageUri != null) {
            takePicture.launch(tempImageUri)
        } else {
            // Không thể tạo URI, báo lỗi cho người dùng
            statusText.text = "Không thể chuẩn bị camera. Vui lòng kiểm tra lại quyền ứng dụng."
            Log.e("HomeActivity", "tempImageUri is null, cannot launch camera.")
        }
    }

    private fun clock(imageUri: Uri) {
        statusText.text = "Đã có ảnh, đang lấy vị trí..."

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED &&
            ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {
            statusText.text = "Lỗi: Mất quyền vị trí."
            return
        }
        fused.lastLocation.addOnSuccessListener { loc: Location? ->
            if (loc == null) {
                statusText.text = "Không lấy được vị trí"
                return@addOnSuccessListener
            }

            statusText.text = "Đang gửi chấm công (Vị trí: ${loc.latitude.format(5)}, ${loc.longitude.format(5)}). Vui lòng chờ..."

            val api = RetrofitClient.retrofit(this).create(ApiService::class.java)

            val stream = contentResolver.openInputStream(imageUri)
            val requestFile = stream!!.readBytes().toRequestBody("image/jpeg".toMediaTypeOrNull())
            stream.close()
            val imagePart = MultipartBody.Part.createFormData("face_image", "face.jpg", requestFile)

            // Part: Text
            val latPart = loc.latitude.toString().toRequestBody("text/plain".toMediaTypeOrNull())
            val lonPart = loc.longitude.toString().toRequestBody("text/plain".toMediaTypeOrNull())
            val locIdPart = selectedWorkLocationId?.toString()?.toRequestBody("text/plain".toMediaTypeOrNull())

            api.clock(latPart, lonPart, locIdPart, imagePart).enqueue(object: Callback<ClockRes> {
                override fun onResponse(call: Call<ClockRes>, response: Response<ClockRes>) {
                    if (response.isSuccessful) {
                        val body = response.body()!!
                        val msg = if (body.within_geofence) "Hợp lệ" else "Ngoài phạm vi"
                        statusText.text = "Đã ${body.type} lúc ${body.timestamp}\nKhoảng cách: ${body.distance_m} m ($msg)"
                        Toast.makeText(this@HomeActivity, "Chấm công ${body.type}", Toast.LENGTH_SHORT).show()
                    } else {
                        val err = try { response.errorBody()?.string() } catch (e: Exception) { null }
                        statusText.text = "Chấm công thất bại: ${response.code()} \nLý do: " + (err ?: "Không rõ")
                    }
                }
                override fun onFailure(call: Call<ClockRes>, t: Throwable) {
                    statusText.text = "Lỗi mạng: ${t.message}"
                }
            })
        }
    }

    fun Double.format(digits: Int) = "%.${digits}f".format(this)

    override fun onSupportNavigateUp(): Boolean {
        onBackPressedDispatcher.onBackPressed()
        return true
    }
}
