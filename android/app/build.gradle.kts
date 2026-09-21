import java.util.Properties

plugins {
    id("com.android.application")
    id("dev.flutter.flutter-gradle-plugin")
}

// Yayin imza anahtari; `android/key.properties` depoda yok, yedekten gelir.
// Yoksa debug anahtarina dusuyoruz ama o APK guncelleme olarak kurulamaz.
val imzaAyarlari = Properties().apply {
    val dosya = rootProject.file("key.properties")
    if (dosya.exists()) dosya.inputStream().use { load(it) }
}
val imzaVar = imzaAyarlari.getProperty("storeFile")?.let {
    rootProject.file(it).exists()
} == true

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
                storeFile = rootProject.file(imzaAyarlari.getProperty("storeFile"))
                storePassword = imzaAyarlari.getProperty("storePassword")
                keyAlias = imzaAyarlari.getProperty("keyAlias")
                keyPassword = imzaAyarlari.getProperty("keyPassword")
            }
        }
    }

    buildTypes {
        release {
            signingConfig = if (imzaVar) {
                signingConfigs.getByName("yayin")
            } else {
                logger.warn(
                    "UYARI: android/key.properties yok. Release APK debug " +
                        "anahtariyla imzalaniyor; telefonlara guncelleme " +
                        "olarak KURULAMAZ. Anahtari yedekten geri koyun.",
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
