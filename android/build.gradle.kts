allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}

// whisper_ggml kendi android/build.gradle'inda compileSdk 34 yaziyor, ama
// bagimliligi ffmpeg_kit_flutter_new_min 35+ istiyor; ikisi carpisinca
// :whisper_ggml:checkDebugAarMetadata patliyor. Eklenti modullerini
// uygulamayla ayni compileSdk'ya cekiyoruz.
//
// Bu blok asagidaki evaluationDependsOn blogundan ONCE gelmeli: o blok
// projeleri degerlendirmeye zorluyor, degerlendirilmis bir projeye
// afterEvaluate eklenemiyor.
subprojects {
    afterEvaluate {
        when (val ext = extensions.findByName("android")) {
            is com.android.build.api.dsl.LibraryExtension ->
                if ((ext.compileSdk ?: 0) < 36) ext.compileSdk = 36
            is com.android.build.api.dsl.ApplicationExtension ->
                if ((ext.compileSdk ?: 0) < 36) ext.compileSdk = 36
            else -> {}
        }
    }
}

subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
