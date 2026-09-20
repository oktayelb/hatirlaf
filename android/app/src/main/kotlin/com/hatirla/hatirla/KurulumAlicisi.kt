package com.hatirla.hatirla

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.pm.PackageInstaller
import android.os.Build
import android.util.Log

/**
 * PackageInstaller'in kurulum sonucunu bildirdigi yer.
 *
 * Sonuc bir PendingIntent ile geldigi icin bir BroadcastReceiver sart;
 * dogrudan geri cagirma (callback) kullanilamiyor. Alici manifest'te
 * tanimli (`exported=false`) - Android 14'ten sonra calisma aninda
 * kaydedilen aliciların `RECEIVER_NOT_EXPORTED` bayragi istemesi ve
 * surec olduruldugunde kaybolmasi yuzunden manifest yolu daha saglam.
 *
 * [dinleyici] uygulama on plandayken [Guncelleyici] tarafindan doldurulur.
 * Bos olabilir: kullanici onay penceresindeyken uygulama olduruldüyse
 * sonucu dinleyen kimse kalmaz. Bu bir sorun degil - bir sonraki acilista
 * [Guncelleyici.surumKodu] zaten kurulumun olup olmadigini soyluyor.
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
        // Kurulum uzaktaki bir telefonda sessizce basarisiz olabiliyor;
        // `adb logcat -s KurulumAlicisi` tek bakista sebebi soylesin.
        Log.i("KurulumAlicisi", "kurulum durumu=$durum mesaj=$mesaj")

        // Sistemin "Kurmak istiyor musunuz?" penceresi.
        //
        // Bu Intent'i buradan baslatmiyoruz: Android 10'dan beri arka plandan
        // ekran acmak engelli. Onun yerine on plandaki Activity'ye veriyoruz,
        // o baslatiyor.
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
