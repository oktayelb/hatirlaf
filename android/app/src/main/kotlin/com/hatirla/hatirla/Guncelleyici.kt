package com.hatirla.hatirla

import android.app.Activity
import android.app.PendingIntent
import android.content.Intent
import android.content.pm.PackageInstaller
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.util.Log
import java.io.File

/**
 * Yeni surumu kuran yerli kopru.
 *
 * Uygulama magazasi olmadigi icin guncellemeyi uygulamanin kendisi kuruyor.
 * `PackageInstaller` oturum API'si kullaniliyor; eski
 * `ACTION_INSTALL_PACKAGE` yolunun aksine bu yol sonucu geri bildiriyor
 * (kullanici vazgecti mi, imza tutmadi mi, yer mi yetmedi) ve FileProvider
 * kurmayi gerektirmiyor.
 *
 * **Sessiz kurulum mumkun degil.** Cihaz sahibi (device owner) olarak
 * saglanmamis bir uygulama, kullaniciya sistemin onay penceresini
 * gostermek zorunda. Yapabildigimiz en iyi sey o pencereye kadar olan her
 * seyi (denetim, indirme, dogrulama) gorunmez halletmek; boylece yasli
 * kullaniciya tek bir "Güncelle" dokunusu kaliyor.
 */
class Guncelleyici(private val activity: Activity) {

    private companion object {
        const val ETIKET = "Guncelleyici"
        const val OTURUM_ADI = "hatirlaf"
    }

    private val anaIsParcacigi = Handler(Looper.getMainLooper())

    /** Kurulum surerken ikinci bir oturum acilmasin. */
    private var kurulumSuruyor = false

    // ---------------------------------------------------------------- surum

    /**
     * Kurulu surumun kodu. `pubspec.yaml`'daki `+N`.
     *
     * Guncelleme karari **yalniz** bu sayiya bakar; surum adi ("1.0.1")
     * kullaniciya gosterilen bir etiketten ibaret.
     */
    fun surumKodu(): Long {
        return try {
            val bilgi = activity.packageManager.getPackageInfo(activity.packageName, 0)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                bilgi.longVersionCode
            } else {
                @Suppress("DEPRECATION")
                bilgi.versionCode.toLong()
            }
        } catch (e: Exception) {
            Log.w(ETIKET, "Surum kodu okunamadi", e)
            0L
        }
    }

    /**
     * Cihazin calistirabilecegi islemci mimarileri, **tercih sirasiyla**.
     *
     * Her mimari icin ayri APK yayinliyoruz: tek parca (universal) APK 60 MB,
     * arm64'e ozel olan 22 MB. Guncelleme her seferinde indirilecegi icin
     * aradaki 38 MB her surumde tekrar tekrar odenen bir bedel olurdu.
     *
     * Liste sirali geliyor: 64 bit bir telefon hem `arm64-v8a` hem
     * `armeabi-v7a` calistirabilir ama ilki tercih edilmeli.
     */
    fun abiler(): List<String> = Build.SUPPORTED_ABIS?.toList() ?: emptyList()

    fun surumAdi(): String {
        return try {
            activity.packageManager
                .getPackageInfo(activity.packageName, 0)
                .versionName ?: "?"
        } catch (e: Exception) {
            "?"
        }
    }

    // ----------------------------------------------------------------- izin

    /**
     * "Bilinmeyen kaynaklardan uygulama yukleme" izni verilmis mi?
     *
     * API 26'dan once bu izin uygulama basina degil cihaz genelindeydi ve
     * uygulamadan okunamiyor; orada `true` varsayip kurulumu deniyoruz,
     * engelliyse sistem zaten kendi penceresini gosteriyor.
     */
    fun kurulumIzniVarMi(): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return true
        return try {
            activity.packageManager.canRequestPackageInstalls()
        } catch (e: Exception) {
            Log.w(ETIKET, "Kurulum izni okunamadi", e)
            false
        }
    }

    /** Kullaniciyi "bu kaynaga izin ver" ekranina goturur. */
    fun kurulumIzniEkraniniAc(): Boolean {
        return try {
            val eylem = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES
            } else {
                Settings.ACTION_SECURITY_SETTINGS
            }
            val intent = Intent(eylem).apply {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    data = Uri.parse("package:${activity.packageName}")
                }
                addFlags(Intent.FLAG_ACTIVITY_NEW_TASK)
            }
            activity.startActivity(intent)
            true
        } catch (e: Exception) {
            Log.w(ETIKET, "Kurulum izni ekrani acilamadi", e)
            false
        }
    }

    // -------------------------------------------------------------- kurulum

    /**
     * [apkYolu]'ndaki APK'yi kurar.
     *
     * [geriBildirim] ana is parcaciginda, tek kelimelik bir sonuc koduyla
     * cagrilir. Ardindan sistemin onay penceresi acilir; kullanici onay
     * verirse surec zaten olduruldugu icin "tamam" sonucu cogu zaman
     * gorulmez - guncellemenin olup olmadigini bir sonraki acilista surum
     * kodu soyler.
     */
    fun kur(apkYolu: String, geriBildirim: (String, String?) -> Unit) {
        if (kurulumSuruyor) {
            geriBildirim("mesgul", null)
            return
        }

        val apk = File(apkYolu)
        if (!apk.isFile || apk.length() <= 0L) {
            geriBildirim("dosyaYok", null)
            return
        }

        kurulumSuruyor = true

        KurulumAlicisi.dinleyici = { durum, mesaj, onay ->
            anaIsParcacigi.post { sonucuIsle(durum, mesaj, onay, geriBildirim) }
        }

        // 45 MB'lik kopyalama ana is parcaciginda yapilirsa arayuz donuyor.
        Thread({ oturumuYaz(apk, geriBildirim) }, "hatirlaf-kurulum").start()
    }

    private fun oturumuYaz(apk: File, geriBildirim: (String, String?) -> Unit) {
        val kurucu = activity.packageManager.packageInstaller
        var oturumNo = -1
        try {
            val ayarlar = PackageInstaller.SessionParams(
                PackageInstaller.SessionParams.MODE_FULL_INSTALL,
            ).apply {
                setAppPackageName(activity.packageName)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    setInstallReason(PackageManager.INSTALL_REASON_USER)
                }
            }

            oturumNo = kurucu.createSession(ayarlar)
            kurucu.openSession(oturumNo).use { oturum ->
                oturum.openWrite(OTURUM_ADI, 0, apk.length()).use { cikis ->
                    apk.inputStream().use { giris -> giris.copyTo(cikis, 64 * 1024) }
                    // fsync olmadan commit, yazilmamis veriyle "bozuk APK"
                    // hatasi verebiliyor.
                    oturum.fsync(cikis)
                }

                // Bilesen acikca veriliyor: eylem adiyla gonderilen ortuk
                // yayinlar Android 8'den beri manifest'teki alicilara
                // ulasmiyor ve kurulum sessizce yarida kaliyor.
                val niyet = Intent(activity, KurulumAlicisi::class.java).apply {
                    action = KurulumAlicisi.EYLEM
                }
                // FLAG_MUTABLE sart: sonucu (durum kodu, onay penceresi)
                // Intent'e sistem dolduruyor.
                val bayraklar = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_MUTABLE
                } else {
                    PendingIntent.FLAG_UPDATE_CURRENT
                }
                val bekleyen = PendingIntent.getBroadcast(
                    activity,
                    oturumNo,
                    niyet,
                    bayraklar,
                )
                oturum.commit(bekleyen.intentSender)
            }
        } catch (e: Exception) {
            Log.w(ETIKET, "Kurulum oturumu acilamadi", e)
            if (oturumNo >= 0) {
                try {
                    kurucu.abandonSession(oturumNo)
                } catch (_: Exception) {
                    // Oturum zaten kapanmis olabilir.
                }
            }
            kurulumSuruyor = false
            KurulumAlicisi.dinleyici = null
            anaIsParcacigi.post {
                val kod = if (e is java.io.IOException) "yer" else "hata"
                geriBildirim(kod, e.message)
            }
        }
    }

    private fun sonucuIsle(
        durum: Int,
        mesaj: String?,
        onay: Intent?,
        geriBildirim: (String, String?) -> Unit,
    ) {
        if (durum == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            if (onay == null) {
                bitir(geriBildirim, "hata", "onay penceresi gelmedi")
                return
            }
            return try {
                // Activity'den baslatiliyor: arka plan kisitlamasina takilmasin.
                activity.startActivity(onay)
                geriBildirim("onayBekleniyor", null)
            } catch (e: Exception) {
                Log.w(ETIKET, "Onay penceresi acilamadi", e)
                bitir(geriBildirim, "hata", e.message)
            }
        }

        val kod = when (durum) {
            PackageInstaller.STATUS_SUCCESS -> "tamam"
            PackageInstaller.STATUS_FAILURE_ABORTED -> "iptal"
            PackageInstaller.STATUS_FAILURE_BLOCKED -> "engellendi"
            PackageInstaller.STATUS_FAILURE_CONFLICT -> "imza"
            PackageInstaller.STATUS_FAILURE_INCOMPATIBLE -> "uyumsuz"
            PackageInstaller.STATUS_FAILURE_INVALID -> "bozuk"
            PackageInstaller.STATUS_FAILURE_STORAGE -> "yer"
            else -> "hata"
        }
        bitir(geriBildirim, kod, mesaj)
    }

    private fun bitir(
        geriBildirim: (String, String?) -> Unit,
        kod: String,
        mesaj: String?,
    ) {
        kurulumSuruyor = false
        KurulumAlicisi.dinleyici = null
        geriBildirim(kod, mesaj)
    }

    /** Activity yok olurken sizinti birakmayalim. */
    fun birak() {
        KurulumAlicisi.dinleyici = null
    }
}
