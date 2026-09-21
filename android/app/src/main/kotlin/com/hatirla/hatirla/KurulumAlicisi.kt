package com.hatirla.hatirla

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.os.Build
import android.util.Log

/**
 * PackageInstaller'in kurulum sonucunu bildirdigi yer. Sonuc PendingIntent
 * ile geldigi icin BroadcastReceiver sart; manifest'te tanimli, cunku
 * calisma aninda kaydedilen alici surec olduruldugunde kaybolur.
 *
 * [dinleyici] bos olabilir (uygulama olduruldüyse); sorun degil, bir
 * sonraki acilista surum kodu zaten sonucu soyluyor.
 */
class KurulumAlicisi : BroadcastReceiver() {

    companion object {
        const val EYLEM = "com.hatirla.hatirla.KURULUM_SONUCU"

        /** (durumKodu, sistemMesaji, onayPenceresi) */
        var dinleyici: ((Int, String?, Intent?) -> Unit)? = null
    }

    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != EYLEM) return

        val durum = intent.getIntExtra(
            PackageInstaller.EXTRA_STATUS,
            PackageInstaller.STATUS_FAILURE,
        )
        val mesaj = intent.getStringExtra(PackageInstaller.EXTRA_STATUS_MESSAGE)
        // `adb logcat -s KurulumAlicisi` sebebi soylesin.
        Log.i("KurulumAlicisi", "kurulum durumu=$durum mesaj=$mesaj")

        // Sistemin onay penceresi. Buradan baslatilmiyor: Android 10'dan
        // beri arka plandan ekran acmak engelli, on plandaki Activity acar.
        val onay: Intent? = if (durum == PackageInstaller.STATUS_PENDING_USER_ACTION) {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                intent.getParcelableExtra(Intent.EXTRA_INTENT, Intent::class.java)
            } else {
                @Suppress("DEPRECATION")
                intent.getParcelableExtra(Intent.EXTRA_INTENT)
            }
        } else {
            null
        }

        dinleyici?.invoke(durum, mesaj, onay)
    }
}
