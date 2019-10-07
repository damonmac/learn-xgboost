package org.familysearch.example;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.*;

public class CreateFeatureVectors2 {

  private static final int[] DATE_FIELDS = {10, 24, 38, 66};
  private static final String OUTPUT_PATH = "data/javaVector";
  private static final String VECTOR_FILE = "data/javaVector";
  private static final String LIBSVM_TRAIN_FILE = "data/javaVector_train.libsvm";
  private static final String LIBSVM_EVAL_FILE = "data/javaVector_eval.libsvm";

  private static final int NUM_EVAL_LINES = 400;

  public static void main(String[] args) throws IOException {
    new CreateFeatureVectors2().run();
  }

  public void run() throws IOException {
    List<Integer> dateFields = Utils.intArrayToList(DATE_FIELDS);
    List<String> lines = Utils.readLines("data/pairs.csv");
    Files.deleteIfExists(Paths.get(OUTPUT_PATH));
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
        else if (i < 9) {
          int sameNames = 0;
          int differentNames = 0;
          List<String> targetNames = Arrays.asList(targetField.split(" "));
          List<String> candidateNames = Arrays.asList(candidateField.split(" "));
          for (String targetName : targetNames) {
            for (String candidateName : candidateNames) {
              if (targetName.equals(candidateName)) {
                sameNames++;
              }
              else {
                differentNames--;
              }
            }
          }
          vectorValues.add(String.valueOf(sameNames > 0 ? sameNames : differentNames));
        }
        else if (dateFields.contains(i)) {
          try {
            int dateDifference = Math.abs(Integer.parseInt(targetField) - Integer.parseInt(candidateField));
            vectorValues.add(String.valueOf(dateDifference < 5 ? 5 - dateDifference : 0));
          }
          catch (NumberFormatException e) {
            vectorValues.add("0");
          }
        }
        else {
          // Basic logic - check if the values are an exact match
          if (targetField.equals(candidateField)) {
            vectorValues.add("1");
          }
          else {
            vectorValues.add("0");
          }

        }
      }
      Path path = Paths.get(OUTPUT_PATH);
      try {
        Files.write(path,
          Collections.singletonList(vectorValues.toString().replaceAll("([,\\[\\]])", "")),
          StandardCharsets.UTF_8,
          Files.exists(path) ? StandardOpenOption.APPEND : StandardOpenOption.CREATE);
      }
      catch (IOException e) {
        System.err.println("Unable to write to file " + OUTPUT_PATH);
        e.printStackTrace();
      }
    }
    createLibSvmFile();
  }


  private void createLibSvmFile() {
    int linesCounted = 0;
    List<String> trainLines = new ArrayList<>();
    List<String> evalLines = new ArrayList<>();
    List<String> inputFile = Utils.readLines(VECTOR_FILE);
    for (String line : inputFile) {
      if (linesCounted < NUM_EVAL_LINES) {
        linesCounted++;
        evalLines.add(line);
      }
      else {
        trainLines.add(line);
      }
    }
    try {
      Files.deleteIfExists(Paths.get(LIBSVM_TRAIN_FILE));
      Files.deleteIfExists(Paths.get(LIBSVM_EVAL_FILE));
    }
    catch (IOException e) {
      e.printStackTrace();
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

