plugins {
    kotlin("jvm")
    id("me.champeau.jmh") version "0.7.3"
}

version = rootProject.version

dependencies {
    jmh(project(":"))
}

kotlin {
    jvmToolchain(21)
}

jmh {
    jmhVersion.set("1.37")
    warmupIterations.set(3)
    iterations.set(5)
    fork.set(1)
    timeUnit.set("s")
    profilers.add("gc")
    resultFormat.set("JSON")
    resultsFile.set(layout.buildDirectory.file("results/jmh/results.json"))
    jvmArgs.add("-Xmx3g")
    if (providers.gradleProperty("kotensor.vectorApi").orNull == "true") {
        jvmArgs.add("--add-modules=jdk.incubator.vector")
    }
}
