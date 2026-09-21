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

// whisper_ggml compileSdk 34 yaziyor ama bagimliligi 35+ istiyor;
// eklenti modullerini uygulamayla ayni compileSdk'ya cekiyoruz.
//
// Bu blok asagidaki evaluationDependsOn blogundan ONCE gelmeli:
// degerlendirilmis bir projeye afterEvaluate eklenemiyor.
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
