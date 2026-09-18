package com.hatirla.hatirla

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

/**
 * Uygulamanin tek yerli koprusu.
 *
 * Buradaki iki islev icin ayri bir izin eklentisi kullanmiyoruz:
 * permission_handler compileSdk 37 istiyor, Google ise o platformu
 * "android-37.0" adiyla yayinladigi icin AGP onu bulamiyor ve derleme
 * kiriliyor. Mikrofon iznini zaten `record` paketi istiyor; geriye sadece
 * su iki soru kaliyor ve ikisi de birkac satir.
 */
class MainActivity : FlutterActivity() {

    private companion object {
        const val KANAL = "hatirla/izin"
    }

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, KANAL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    // Kullaniciyi uygulamanin izin ekranina goturur.
                    "ayarlariAc" -> {
                        try {
                            val intent = Intent(
                                Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                                Uri.fromParts("package", packageName, null),
                            ).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
                            startActivity(intent)
                            result.success(true)
                        } catch (e: Exception) {
                            result.success(false)
                        }
                    }

                    // "Bir daha sorma" secilmis mi?
                    //
                    // shouldShowRequestPermissionRationale() bir reddedilmeden
                    // SONRA false donuyorsa sistem artik izin penceresini
                    // acmayacak demektir. Bu yuzden yalnizca izin istegi
                    // reddedildikten sonra cagrilmali; ilk istekten once de
                    // false donduğu icin yanlis pozitif verir.
                    "kaliciReddedildiMi" -> {
                        try {
                            val sorulabilir = shouldShowRequestPermissionRationale(
                                Manifest.permission.RECORD_AUDIO,
                            )
                            result.success(!sorulabilir)
                        } catch (e: Exception) {
                            result.success(false)
                        }
                    }

                    else -> result.notImplemented()
                }
            }
    }
}
