plugins {
    kotlin("multiplatform") version "2.0.21"
    kotlin("plugin.serialization") version "2.0.21"
    id("com.android.library") version "8.2.0"
    id("maven-publish")
    id("signing")
}

group = "com.daemoniorum"
version = "0.1.0"

repositories {
    mavenCentral()
    google()
}

kotlin {
    // JVM target (Desktop, Server)
    jvm {
        compilations.all {
            kotlinOptions.jvmTarget = "17"
        }
        testRuns["test"].executionTask.configure {
            useJUnitPlatform()
        }
    }

    // Android target
    androidTarget {
        compilations.all {
            kotlinOptions.jvmTarget = "17"
        }
        publishLibraryVariants("release")
    }

    // iOS targets
    iosX64()
    iosArm64()
    iosSimulatorArm64()

    // macOS targets
    macosX64()
    macosArm64()

    // Linux target
    linuxX64()

    // Windows target
    mingwX64()

    // JavaScript/Browser target
    js(IR) {
        browser {
            testTask {
                useKarma {
                    useChromeHeadless()
                }
            }
        }
        nodejs()
        binaries.executable()
    }

    // WebAssembly target
    @OptIn(org.jetbrains.kotlin.gradle.targets.js.dsl.ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }

    sourceSets {
        val commonMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.1")
                implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.7.3")
                implementation("org.jetbrains.kotlinx:kotlinx-datetime:0.6.1")
            }
        }

        val commonTest by getting {
            dependencies {
                implementation(kotlin("test"))
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.1")
            }
        }

        val jvmMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-jvm:1.8.1")
            }
        }

        val jvmTest by getting {
            dependencies {
                implementation("org.junit.jupiter:junit-jupiter:5.10.0")
            }
        }

        val androidMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
            }
        }

        val iosX64Main by getting
        val iosArm64Main by getting
        val iosSimulatorArm64Main by getting
        val iosMain by creating {
            dependsOn(commonMain)
            iosX64Main.dependsOn(this)
            iosArm64Main.dependsOn(this)
            iosSimulatorArm64Main.dependsOn(this)
        }

        val nativeMain by creating {
            dependsOn(commonMain)
        }

        val jsMain by getting {
            dependencies {
                implementation("org.jetbrains.kotlinx:kotlinx-coroutines-core-js:1.8.1")
            }
        }
    }
}

android {
    namespace = "com.daemoniorum.sigil"
    compileSdk = 34
    defaultConfig {
        minSdk = 24
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
}

// Create javadoc jar for Maven Central
val javadocJar by tasks.registering(Jar::class) {
    archiveClassifier.set("javadoc")
}

// Publishing configuration for Maven Central
publishing {
    publications.withType<MavenPublication> {
        artifact(javadocJar)

        pom {
            name.set("Sigil")
            description.set("Polysynthetic programming language with epistemic types for AI")
            url.set("https://github.com/Daemoniorum-LLC/sigil-lang")

            licenses {
                license {
                    name.set("Daemoniorum Source-Available License")
                    url.set("https://github.com/Daemoniorum-LLC/sigil-lang/blob/main/LICENSE")
                }
            }

            developers {
                developer {
                    id.set("lilith")
                    name.set("Lilith Crook")
                    email.set("lilith@daemoniorum.com")
                    organization.set("Daemoniorum, LLC")
                    organizationUrl.set("https://www.daemoniorum.com")
                }
            }

            scm {
                url.set("https://github.com/Daemoniorum-LLC/sigil-lang")
                connection.set("scm:git:git://github.com/Daemoniorum-LLC/sigil-lang.git")
                developerConnection.set("scm:git:ssh://github.com/Daemoniorum-LLC/sigil-lang.git")
            }
        }
    }

    repositories {
        maven {
            name = "MavenCentral"
            url = uri("https://s01.oss.sonatype.org/service/local/staging/deploy/maven2/")

            credentials {
                username = System.getenv("MAVEN_USERNAME") ?: project.findProperty("mavenCentralUsername") as String?
                password = System.getenv("MAVEN_PASSWORD") ?: project.findProperty("mavenCentralPassword") as String?
            }
        }

        maven {
            name = "GitHubPackages"
            url = uri("https://maven.pkg.github.com/Daemoniorum-LLC/sigil-lang")

            credentials {
                username = System.getenv("GITHUB_ACTOR") ?: project.findProperty("gpr.user") as String?
                password = System.getenv("GITHUB_TOKEN") ?: project.findProperty("gpr.token") as String?
            }
        }
    }
}

// Signing for Maven Central
signing {
    val signingKey = System.getenv("GPG_SIGNING_KEY") ?: project.findProperty("signing.key") as String?
    val signingPassword = System.getenv("GPG_SIGNING_PASSWORD") ?: project.findProperty("signing.password") as String?

    if (signingKey != null && signingPassword != null) {
        useInMemoryPgpKeys(signingKey, signingPassword)
        sign(publishing.publications)
    }
}

// Ensure signing happens before publishing
tasks.withType<PublishToMavenRepository>().configureEach {
    dependsOn(tasks.withType<Sign>())
}
