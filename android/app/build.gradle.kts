import java.io.File

plugins {
    id("com.android.application")
    id("dev.flutter.flutter-gradle-plugin")
}

// Yayin imza sirlari depo kokundeki `.env` dosyasinda; depoda yok,
// yedekten gelir. Bu degerler APK'yi IMZALAMAKTA kullanilir, APK'nin
// icine girmez -- derlemeye giren sirlar yedek.json'dadir.
//
// Once .env, sonra gercek ortam degiskenleri: CI ya da kabuktan
// gecirmek isteyen dosya olusturmak zorunda kalmasin.
val depoKoku = rootProject.projectDir.parentFile

val envDegerleri: Map<String, String> = run {
    val dosya = File(depoKoku, ".env")
    if (!dosya.exists()) {
        emptyMap()
    } else {
        dosya.readLines()
            .map { it.trim() }
            .filter { it.isNotEmpty() && !it.startsWith("#") && it.contains("=") }
            .associate { satir ->
                val ad = satir.substringBefore("=").trim()
                // Tirnakli yazilmis degerler de kabul edilsin.
                val deger = satir.substringAfter("=").trim()
                    .removeSurrounding("\"")
                    .removeSurrounding("'")
                ad to deger
            }
    }
}

val sir: (String) -> String? = { ad ->
    envDegerleri[ad]?.takeIf { it.isNotEmpty() }
        ?: System.getenv(ad)?.takeIf { it.isNotEmpty() }
}

// Yol depo koküne gore; mutlak yol da calissin.
val imzaDosyasi: File? = sir("ANDROID_KEYSTORE")?.let { yol ->
    val f = File(yol)
    (if (f.isAbsolute) f else File(depoKoku, yol)).takeIf { it.exists() }
}

val imzaVar = imzaDosyasi != null &&
    sir("ANDROID_KEYSTORE_PASSWORD") != null &&
    sir("ANDROID_KEY_ALIAS") != null &&
    sir("ANDROID_KEY_PASSWORD") != null

android {
    namespace = "com.hatirla.hatirla"
    compileSdk = 36

    // whisper_ggml bu NDK surumunu kullaniyor; hepsi ayni olsun.
    ndkVersion = "29.0.13113456"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
        isCoreLibraryDesugaringEnabled = true
    }

    defaultConfig {
        applicationId = "com.hatirla.hatirla"
        // ffmpeg_kit ve record 24+ istiyor
        minSdk = 24
        targetSdk = 36
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    signingConfigs {
        if (imzaVar) {
            create("yayin") {
                storeFile = imzaDosyasi
                storePassword = sir("ANDROID_KEYSTORE_PASSWORD")
                keyAlias = sir("ANDROID_KEY_ALIAS")
                keyPassword = sir("ANDROID_KEY_PASSWORD")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = if (imzaVar) {
                signingConfigs.getByName("yayin")
            } else {
                logger.warn(
                    "UYARI: .env icindeki imza ayarlari eksik (ANDROID_KEYSTORE, " +
                        "ANDROID_KEYSTORE_PASSWORD, ANDROID_KEY_ALIAS, " +
                        "ANDROID_KEY_PASSWORD). Release APK debug anahtariyla " +
                        "imzalaniyor; telefonlara guncelleme olarak KURULAMAZ.",
                )
                signingConfigs.getByName("debug")
            }
            // whisper.cpp + ffmpeg native kutuphaneleri R8 ile ugrasmasin
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }
}

kotlin {
    compilerOptions {
        jvmTarget = org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17
    }
}

flutter {
    source = "../.."
}

dependencies {
    coreLibraryDesugaring("com.android.tools:desugar_jdk_libs:2.1.5")
}
