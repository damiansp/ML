name := "02_Breeze"

version := "0.1"

scalaVersion := "2.12.8"

libraryDependencies ++= Seq(
  "org.scalanlp" %% "breeze" % "1.0-RC2",
  "org.scalanlp" %% "breeze-natives" % "1.0-RC2",
  "org.slf4j" % "slf4j-simple" % "1.7.25" % Test)