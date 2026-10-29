plugins {
    kotlin("jvm") version "2.3.20"
    `maven-publish`
}

group = "com.github.TSPstudio"
version = "0.1.0"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
}

kotlin {
    jvmToolchain(21)
}

java {
    withSourcesJar()
}

publishing {
    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            groupId = "com.github.TSPstudio"
            artifactId = "kotensor"
            version = "0.1.0"

            pom {
                name.set("Kotensor")
                description.set("Pure Kotlin Autograd and Generative Transformer Engine")
                url.set("https://github.com/TSPstudio/kotensor")
                licenses {
                    license {
                        name.set("The Apache License, Version 2.0")
                        url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
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
                    connection.set("scm:git:git://github.com/TSPstudio/kotensor.git")
                    developerConnection.set("scm:git:ssh://github.com:TSPstudio/kotensor.git")
                    url.set("https://github.com/TSPstudio/kotensor")
                }
            }
        }
    }
}

tasks.test {
    useJUnitPlatform()
}