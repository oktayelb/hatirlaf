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
 * Uygulamanin tek yerli koprusu. permission_handler yerine elde yazildi:
 * o compileSdk 37 istiyor, Google platformu "android-37.0" adiyla
 * yayinladigi icin AGP bulamiyor ve derleme kiriliyor.
 */
class MainActivity : FlutterActivity() {

    private companion object {
        const val IZIN_KANALI = "hatirla/izin"
        const val GUNCELLEME_KANALI = "hatirla/guncelleme"
        const val AG_KANALI = "hatirla/ag"
        const val KAYIT_KANALI = "hatirla/kayit"
    }

    private var guncelleyici: Guncelleyici? = null

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        val mesajci = flutterEngine.dartExecutor.binaryMessenger

        MethodChannel(mesajci, IZIN_KANALI).setMethodCallHandler { call, result ->
            when (call.method) {
                // Uygulamanin izin ekranina goturur.
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

                // "Bir daha sorma" secilmis mi? Yalnizca bir reddedilmeden
                // SONRA cagrilmali: shouldShowRequestPermissionRationale()
                // ilk istekten once de false doner, yani yanlis pozitif.
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
                        "apkYolu" to g.kuruluApkYolu(),
                    ),
                )

                "kurulumIzniVarMi" -> result.success(g.kurulumIzniVarMi())

                "kurulumIzniEkraniniAc" -> result.success(g.kurulumIzniEkraniniAc())

                // Kurulum birden fazla yanit uretiyor (once
                // "onayBekleniyor", sonra sonuc). MethodChannel bir cagriya
                // iki kez yanit veremedigi icin ters yonde itiyoruz.
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

        // Kayit boyunca surecin oldurulmesini engelleyen on plan servisi.
        MethodChannel(mesajci, KAYIT_KANALI).setMethodCallHandler { call, result ->
            when (call.method) {
                "basla" -> result.success(KayitServisi.basla(applicationContext))
                "bitir" -> result.success(KayitServisi.bitir(applicationContext))
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
