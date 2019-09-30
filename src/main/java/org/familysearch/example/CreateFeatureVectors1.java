package org.familysearch.example;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class CreateFeatureVectors1 {

  private static final String OUTPUT_PATH = "data/javaVector";
  private static final String VECTOR_FILE = "data/javaVector";
  private static final String LIBSVM_TRAIN_FILE = "data/javaVector_train.libsvm";
  private static final String LIBSVM_EVAL_FILE = "data/javaVector_eval.libsvm";

  public static void main(String[] args) {
    new CreateFeatureVectors1().run();
  }

  public void run() {
    List<String> lines = Utils.readLines("data/pairs.csv");
    for (String line : lines) {
      List<String> vectorValues = new ArrayList<>();
      List<String> fields = new ArrayList<>(Arrays.asList(line.split(",")));
      // Add the match value to the result vector
      vectorValues.add(fields.get(0));
      // Remove the first value that identifies the pair as a match or not
      fields.remove(0);
      for (int i = 0; i < fields.size() - 1; i = i + 2) {
        String targetField = fields.get(i);
        String candidateField = fields.get(i + 1);
        if (targetField.isEmpty() || candidateField.isEmpty()) {
          vectorValues.add("0");
        }
        else if (targetField.equals(candidateField)) {
          vectorValues.add("1");
        }
        else {
          vectorValues.add("0");
        }
      }
      Path path = Paths.get(OUTPUT_PATH);
      try {
        Files.write(path,
          Collections.singletonList(vectorValues.toString().replaceAll("([,\\[\\]])", "")),
          StandardCharsets.UTF_8,
          Files.exists(path) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE);
        createLibSvmFile();
      }
      catch (IOException e) {
        System.err.println("Unable to write to file " + OUTPUT_PATH);
        e.printStackTrace();
      }
    }

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
        for (int i = 1; i < features.length; i++) {
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

