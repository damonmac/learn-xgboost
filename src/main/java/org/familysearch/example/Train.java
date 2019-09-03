package org.familysearch.example;

import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Train {

  private static final String VECTOR_FILE = "data/javaVector";
  private static final String LIBSVM_TRAIN_FILE = "data/javaVector_train.libsvm";
  private static final String LIBSVM_EVAL_FILE = "data/javaVector_eval.libsvm";
  private static final String OUTPUT_MODEL_FILE = "xgboost.model";
  private static final int TRAINING_ITERATIONS = 1000;

  public static void main(String[] args) {
    try {
      new Train().run();
    }
    catch (XGBoostError xgBoostError) {
      System.err.println("Unable to train! Error occurred.");
      xgBoostError.printStackTrace();
    }
  }

  public void run() throws XGBoostError {
    createLibSvmFile();
    DMatrix trainMatrix = new DMatrix(LIBSVM_TRAIN_FILE);
    DMatrix evalMatrix = new DMatrix(LIBSVM_EVAL_FILE);

    Map<String, Object> params = new HashMap<>();

    // Add any other desired params here....
    params.put("eta", 0.1);
    params.put("max_depth", 4);
    params.put("gamma", 0);
    params.put("subsample", 1);
    params.put("colsample_bytree", 1);
    params.put("objective", "binary:logistic");

    Map<String, DMatrix> watches = new HashMap<>();
    watches.put("train", trainMatrix);
    watches.put("eval", evalMatrix);

    Booster booster = XGBoost.train(trainMatrix, params, TRAINING_ITERATIONS, watches, null, null);
    booster.saveModel(OUTPUT_MODEL_FILE);
  }

  private void createLibSvmFile() {
    List<String> trainLines = new ArrayList<>();
    List<String> evalLines = new ArrayList<>();
    List<String> inputFile = Utils.readLines(VECTOR_FILE);
    for (String line : inputFile) {
      if (Math.random() > 0.15) {
        trainLines.add(line);
      }
      else {
        evalLines.add(line);
      }
    }
    writeLibSvmLines(trainLines, LIBSVM_TRAIN_FILE);
    writeLibSvmLines(evalLines, LIBSVM_EVAL_FILE);
  }

  private void writeLibSvmLines(List<String> inputFile, String libsvmEvalFile) {
    Path libSvmEvalFile = Paths.get(libsvmEvalFile);
    try (BufferedWriter writer = Files.newBufferedWriter(libSvmEvalFile)) {
      StringJoiner sj = new StringJoiner(" ");
      for (String line : inputFile) {
        String[] features = line.split(" ");
        sj.add(features[0]);
        for (int i=1; i<features.length; i++) {
          String featureScore = features[i];
          if (!"0".equals(featureScore)) {
            sj.add(i + ":" + featureScore);
          }
        }
        sj.add("\n");
      }
      writer.write(sj.toString());
    }
    catch (IOException e) {
      System.err.println("Unable to write to file " + libsvmEvalFile);
      e.printStackTrace();
    }
  }

}
