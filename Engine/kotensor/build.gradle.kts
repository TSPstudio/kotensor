import com.vanniktech.maven.publish.JavadocJar
import com.vanniktech.maven.publish.KotlinJvm
import com.vanniktech.maven.publish.SourcesJar
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.3.20"
    id("org.jetbrains.dokka") version "2.2.0"
    id("com.vanniktech.maven.publish") version "0.37.0"
}

group = "io.github.tspstudio"
version = "1.0.0"

dependencies {
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
        freeCompilerArgs.add("-Xjdk-release=17")
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.release.set(17)
}

tasks.test {
    useJUnitPlatform()
}

tasks.withType<Jar>().configureEach {
    metaInf {
        from(file("../../LICENSE"), file("../../NOTICE"))
    }
}

dokka {
    moduleName.set("Kotensor")
    dokkaSourceSets.main {
        includes.from("Module.md")
        sourceLink {
            localDirectory.set(file("src/main/kotlin"))
            remoteUrl("https://github.com/TSPstudio/kotensor/tree/main/Engine/kotensor/src/main/kotlin")
            remoteLineSuffix.set("#L")
        }
    }
    pluginsConfiguration.html {
        footerMessage.set("Kotensor · pure Kotlin autograd and Transformer engine")
    }
}

mavenPublishing {
    configure(KotlinJvm(javadocJar = JavadocJar.Dokka("dokkaGeneratePublicationHtml"), sourcesJar = SourcesJar.Sources()))
    publishToMavenCentral(automaticRelease = true)
    coordinates("io.github.tspstudio", "kotensor", version.toString())

    pom {
        name.set("Kotensor")
        description.set("Pure Kotlin autograd and generative Transformer engine: training, LoRA, KV-cached decoding, INT8/INT4 quantization and Hugging Face import with zero native dependencies")
        url.set("https://github.com/TSPstudio/kotensor")
        inceptionYear.set("2026")
        licenses {
            license {
                name.set("The Apache License, Version 2.0")
                url.set("https://www.apache.org/licenses/LICENSE-2.0.txt")
                distribution.set("repo")
            }
        }
        developers {
            developer {
                id.set("TSPstudio")
                name.set("TSPstudio")
                url.set("https://github.com/TSPstudio")
            }
        }
        scm {
            url.set("https://github.com/TSPstudio/kotensor")
            connection.set("scm:git:git://github.com/TSPstudio/kotensor.git")
            developerConnection.set("scm:git:ssh://git@github.com/TSPstudio/kotensor.git")
        }
    }
}
