package com.hatirla.hatirla

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import android.util.Log

/**
 * Kayit surerken calisan on plan servisi.
 *
 * Tek isi var: kayit boyunca surecin oldurulmesini engellemek. Android 12
 * ve sonrasinda arka plana alinan bir uygulamanin mikrofonu birakmasi ve
 * surecin toplanmasi normaldir -- ama burada kaybedilen sey bir dosya
 * degil, bir daha anlatilmayacak bir hatira. wakelock ekrani acik tutuyor,
 * sureci tutmuyor; onu ancak on plan servisi yapabiliyor.
 *
 * Bildirim gorunmeyebilir: Android 13'ten beri bildirim gostermek ayri bir
 * izin istiyor ve bu uygulama onu bilerek istemiyor (yasli kullaniciya bir
 * izin penceresi daha cikarmamak icin). Izin yoksa bildirim gizli kalir,
 * servis yine calisir -- korunan sey bildirim degil, surectir.
 */
class KayitServisi : Service() {

    companion object {
        private const val ETIKET = "KayitServisi"
        private const val KANAL = "kayit"
        private const val BILDIRIM_NO = 1881

        /** Kayit basladiginda; uygulama on plandayken cagrilmali. */
        fun basla(context: Context): Boolean {
            return try {
                val niyet = Intent(context, KayitServisi::class.java)
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
                    context.startForegroundService(niyet)
                } else {
                    context.startService(niyet)
                }
                true
            } catch (e: Exception) {
                // Servis baslamadiysa kayit yine de alinir; yalnizca surec
                // korumasi yok demektir.
                Log.w(ETIKET, "On plan servisi baslatilamadi", e)
                false
            }
        }

        fun bitir(context: Context): Boolean {
            return try {
                context.stopService(Intent(context, KayitServisi::class.java))
                true
            } catch (e: Exception) {
                Log.w(ETIKET, "On plan servisi durdurulamadi", e)
                false
            }
        }
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        try {
            kanaliAc()
            // Tur bildirilmezse Android 14 servisi oldurur.
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                startForeground(
                    BILDIRIM_NO,
                    bildirim(),
                    ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE,
                )
            } else {
                startForeground(BILDIRIM_NO, bildirim())
            }
        } catch (e: Exception) {
            Log.w(ETIKET, "On plana gecilemedi", e)
            stopSelf()
        }
        // Sistem sureci yine de oldururse servis kendiliginden dirilmesin:
        // mikrofonu tutan kayit artik yok.
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                stopForeground(STOP_FOREGROUND_REMOVE)
            } else {
                @Suppress("DEPRECATION")
                stopForeground(true)
            }
        } catch (e: Exception) {
            Log.w(ETIKET, "On plandan cikilamadi", e)
        }
        super.onDestroy()
    }

    private fun kanaliAc() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val yonetici = getSystemService(NotificationManager::class.java) ?: return
        if (yonetici.getNotificationChannel(KANAL) != null) return
        // Dusuk onem: ses cikarmasin, kayit sirasinda titremesin.
        val kanal = NotificationChannel(
            KANAL,
            "Ses kaydı",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Kayıt sürerken görünen bildirim"
            setShowBadge(false)
            enableVibration(false)
        }
        yonetici.createNotificationChannel(kanal)
    }

    private fun bildirim(): Notification {
        val bayraklar = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        } else {
            PendingIntent.FLAG_UPDATE_CURRENT
        }
        val ac = PendingIntent.getActivity(
            this,
            0,
            Intent(this, MainActivity::class.java)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK),
            bayraklar,
        )

        val kurucu = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, KANAL)
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
        }

        return kurucu
            .setContentTitle("Hatıra kaydediliyor")
            .setContentText("Kayıt sürüyor, uygulamayı kapatmayın.")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(ac)
            .setOngoing(true)
            .build()
    }
}
