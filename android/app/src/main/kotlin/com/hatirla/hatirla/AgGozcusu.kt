package com.hatirla.hatirla

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.os.Handler
import android.os.Looper
import android.util.Log
import io.flutter.plugin.common.EventChannel

/**
 * Ag durumunu Dart tarafina akitan gozcu. `yok` / `sayacli` / `serbest`
 * yayar. connectivity_plus yerine elde yazildi: bize baglanti turu degil
 * sayacli mi bilgisi lazim, ve her yeni eklenti bir derleme riski.
 *
 * `NET_CAPABILITY_VALIDATED` da araniyor: Wi-Fi'ye bagli olmak internete
 * cikabilmek demek degil (kafe giris sayfalari, modem acik hat yok).
 */
class AgGozcusu(private val context: Context) : EventChannel.StreamHandler {

    private companion object {
        const val ETIKET = "AgGozcusu"
    }

    private val anaIsParcacigi = Handler(Looper.getMainLooper())
    private var yayinci: EventChannel.EventSink? = null
    private var geriCagirma: ConnectivityManager.NetworkCallback? = null
    private var sonDeger: String? = null

    private val yonetici: ConnectivityManager?
        get() = context.getSystemService(Context.CONNECTIVITY_SERVICE) as? ConnectivityManager

    override fun onListen(arguments: Any?, events: EventChannel.EventSink?) {
        yayinci = events
        sonDeger = null

        val cm = yonetici
        if (cm == null) {
            gonder("yok")
            return
        }

        // Sistem yalnizca degisiklikte atesliyor; ilk degeri kendimiz
        // bildirmezsek uygulama acilista agi "yok" sanir.
        gonder(durumuOku(cm))

        val gc = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(
                network: Network,
                ozellikler: NetworkCapabilities,
            ) {
                gonder(ozellikleriCevir(ozellikler))
            }

            override fun onLost(network: Network) {
                // Baska bir ag devralmis olabilir; yeniden okuyoruz.
                gonder(durumuOku(yonetici))
            }

            override fun onUnavailable() {
                gonder("yok")
            }
        }

        try {
            cm.registerDefaultNetworkCallback(gc)
            geriCagirma = gc
        } catch (e: Exception) {
            Log.w(ETIKET, "Ag dinleyicisi kurulamadi", e)
        }
    }

    override fun onCancel(arguments: Any?) {
        geriCagirma?.let { gc ->
            try {
                yonetici?.unregisterNetworkCallback(gc)
            } catch (e: Exception) {
                // Zaten kayitli degilse sistem firlatiyor; onemsiz.
                Log.w(ETIKET, "Ag dinleyicisi birakilamadi", e)
            }
        }
        geriCagirma = null
        yayinci = null
        sonDeger = null
    }

    private fun durumuOku(cm: ConnectivityManager?): String {
        if (cm == null) return "yok"
        return try {
            val ag = cm.activeNetwork ?: return "yok"
            val ozellikler = cm.getNetworkCapabilities(ag) ?: return "yok"
            ozellikleriCevir(ozellikler)
        } catch (e: Exception) {
            Log.w(ETIKET, "Ag durumu okunamadi", e)
            "yok"
        }
    }

    private fun ozellikleriCevir(ozellikler: NetworkCapabilities): String {
        val internetVar = ozellikler.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
        val dogrulandi = ozellikler.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED)
        if (!internetVar || !dogrulandi) return "yok"

        val sayacsiz = ozellikler.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED)
        return if (sayacsiz) "serbest" else "sayacli"
    }

    private fun gonder(deger: String) {
        // Sistem ayni yetenekleri saniyede birkac kez bildirebiliyor.
        if (deger == sonDeger) return
        sonDeger = deger
        anaIsParcacigi.post { yayinci?.success(deger) }
    }
}
