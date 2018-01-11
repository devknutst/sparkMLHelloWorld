import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD



object SimpleApp {
  def main(args: Array[String]) {

    val a: Int = 2
    val logFile = "README.md"
    val sc = new SparkContext("local", "Simple App", "~/oss/spark")

    val  data: RDD[String]  = sc.textFile("src/resources/Qualitative_Bankruptcy.data.txt")
    data.count()

    val parsedData: RDD[LabeledPoint] = data.map{line =>
      val parts = line.split(",")
      LabeledPoint(getDoubleValue(parts(6)), Vectors.dense(parts.slice(0,6).map(x => getDoubleValue(x))))
    }

    println(parsedData.take(10).mkString("\n"))

    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.6, 0.4), seed = 11L)
    val trainingData: RDD[LabeledPoint] = splits(0)
    val testData: RDD[LabeledPoint] = splits(1)

    // Train the model
    val model: LogisticRegressionModel  = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)

    // Evaluate model on training examples and compute training error
    val labelAndPreds: RDD[(Double, Double)] = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }

    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count
    println("Training Error = " + trainErr)

  }


  def getDoubleValue( input:String ) : Double = {
    var result:Double = 0.0
    if (input == "P")  result = 3.0
    if (input == "A")  result = 2.0
    if (input == "N")  result = 1.0
    if (input == "NB") result = 1.0
    if (input == "B")  result = 0.0

    result
  }
}




