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
 * Aga baglanma durumunu Dart tarafina akitan gozcu.
 *
 * Neden ayri bir eklenti (connectivity_plus) degil? Iki sebep: bu proje
 * compileSdk/NDK surumlerini elle sabitledigi icin her yeni eklenti bir
 * derleme riski (bkz. MainActivity'deki permission_handler notu), ve bize
 * baglanti *turu* degil **sayacli mi** bilgisi lazim - paylasilan mobil
 * baglantiyi Wi-Fi sanip yasli kullanicinin faturasina 45 MB yazmayalim.
 *
 * Yayilan degerler:
 *  - `yok`      : internet yok (ya da henuz dogrulanmadi)
 *  - `sayacli`  : internet var ama sayacli (mobil veri, sinirli hotspot)
 *  - `serbest`  : internet var ve sayacsiz (Wi-Fi, ethernet)
 *
 * `NET_CAPABILITY_VALIDATED` da araniyor: telefonun Wi-Fi'ye bagli olmasi
 * internete cikabildigi anlamina gelmiyor (otel/kafe giris sayfalari,
 * modem acik ama hat yok). Dogrulanmamis agda indirmeye baslamak yarim
 * kalan dosyalardan baska bir sey uretmiyor.
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

        // Ilk deger: dinlemeye baslar baslamaz mevcut durumu bildir.
        // Sistem geri cagirmasi yalnizca *degisiklikte* atesleniyor; bunu
        // atlarsak uygulama acilista agi "yok" sanir.
        gonder(durumuOku(cm))

        val gc = object : ConnectivityManager.NetworkCallback() {
            override fun onCapabilitiesChanged(
                network: Network,
                ozellikler: NetworkCapabilities,
            ) {
                gonder(ozellikleriCevir(ozellikler))
            }

            override fun onLost(network: Network) {
                // Baska bir ag devralmis olabilir (Wi-Fi > mobil); tek bir
                // agin kaybini "internet yok" saymak yerine yeniden okuyoruz.
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
        // Sistem ayni yetenekleri saniyede birkac kez bildirebiliyor;
        // degismediyse Dart tarafini uyandirmaya gerek yok.
        if (deger == sonDeger) return
        sonDeger = deger
        anaIsParcacigi.post { yayinci?.success(deger) }
    }
}
