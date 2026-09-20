package com.hatirla.hatirla

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.EventChannel
import io.flutter.plugin.common.MethodChannel

/**
 * Uygulamanin tek yerli koprusu.
 *
 * Buradaki islevler icin ayri eklentiler kullanmiyoruz:
 * permission_handler compileSdk 37 istiyor, Google ise o platformu
 * "android-37.0" adiyla yayinladigi icin AGP onu bulamiyor ve derleme
 * kiriliyor. Mikrofon iznini zaten `record` paketi istiyor; geriye kalan
 * birkac soru da birkac satir.
 */
class MainActivity : FlutterActivity() {

    private companion object {
        const val IZIN_KANALI = "hatirla/izin"
        const val GUNCELLEME_KANALI = "hatirla/guncelleme"
        const val AG_KANALI = "hatirla/ag"
    }

    private var guncelleyici: Guncelleyici? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val mesajci = flutterEngine.dartExecutor.binaryMessenger

        MethodChannel(mesajci, IZIN_KANALI).setMethodCallHandler { call, result ->
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

        val g = Guncelleyici(this)
        guncelleyici = g

        val guncellemeKanali = MethodChannel(mesajci, GUNCELLEME_KANALI)
        guncellemeKanali.setMethodCallHandler { call, result ->
            when (call.method) {
                "surum" -> result.success(
                    mapOf(
                        "surumKodu" to g.surumKodu(),
                        "surumAdi" to g.surumAdi(),
                        "abiler" to g.abiler(),
                    ),
                )

                "kurulumIzniVarMi" -> result.success(g.kurulumIzniVarMi())

                "kurulumIzniEkraniniAc" -> result.success(g.kurulumIzniEkraniniAc())

                // Kurulum tek bir yanit uretmiyor: once "onayBekleniyor",
                // sonra kullanici karar verince "iptal"/"imza"/"tamam".
                // MethodChannel bir cagriya ikinci kez yanit veremedigi icin
                // sonuclari Dart'a ters yonde ("kurulumSonucu") itiyoruz.
                "kur" -> {
                    val yol = call.argument<String>("apkYolu")
                    if (yol.isNullOrBlank()) {
                        result.success(false)
                        guncellemeKanali.invokeMethod(
                            "kurulumSonucu",
                            mapOf("sonuc" to "dosyaYok", "mesaj" to null),
                        )
                    } else {
                        result.success(true)
                        g.kur(yol) { kod, mesaj ->
                            guncellemeKanali.invokeMethod(
                                "kurulumSonucu",
                                mapOf("sonuc" to kod, "mesaj" to mesaj),
                            )
                        }
                    }
                }

                else -> result.notImplemented()
            }
        }

        EventChannel(mesajci, AG_KANALI).setStreamHandler(AgGozcusu(applicationContext))
    }

    override fun onDestroy() {
        guncelleyici?.birak()
        guncelleyici = null
        super.onDestroy()
    }
}
