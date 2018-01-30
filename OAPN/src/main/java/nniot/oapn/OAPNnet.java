/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nniot.oapn;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;

/**
 *
 * @author nick
 * @author bryson
 */
public class OAPNnet {

    static final int numAttributes = 10;
    static final int numTestRecords = 100;
    static final int numTrainingRecords = 50;
    static final String testDataFN = ""; //this file will not have classifications
    static final String weightsFN = ""; //this file will not have classifications
    
    static final String trainingDataFN = ""; // this file will already have classifications
    static Boolean isTraining = false;
    //This map is shared among all stages
    static HashMap<Pipe<MessageSchemaDynamic>, Float> weightsMap;

    private static Appendable target;
    static Pipe<MessageSchemaDynamic>[][] fromA;

    static Pipe<MessageSchemaDynamic>[][] fromB;

    static Pipe<MessageSchemaDynamic>[] fromC;
    static Pipe<MessageSchemaDynamic>[] prevA;

    public static void main(String[] args) throws FileNotFoundException {
        String[][] trainingData = new String[numTestRecords][numAttributes + 1];
        String[][] testingData = new String[numTrainingRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];

        trainingData = readInData(trainingData, trainingDataFN);
        testingData = readInData(testingData, testDataFN);

        target = null;

        GraphManager gm = new GraphManager();
        GraphManager.addDefaultNota(gm, GraphManager.SCHEDULE_RATE, 20_000);

        if (isTraining) {
            buildVisualNeuralNet(gm, trainingData);
        } else {
            buildVisualNeuralNet(gm, testingData);
        }

        gm.enableTelemetry(8089);

        StageScheduler.defaultScheduler(gm).startup();
    }

    public static String[][] readInData(String[][] data, String fn) {
        String line = null;
        try {
            int i = 0;
            FileReader fileReader = new FileReader(fn);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while ((line = bufferedReader.readLine()) != null) {
                data[i] = line.split(",");
                i++;
            }
            bufferedReader.close();
        } catch (FileNotFoundException ex) {
            System.out.println("Unable to open file:" + fn);
        } catch (IOException ex) {
            System.out.println("IOException while reading file:" + fn);
        }
        return data;
    }

    //Incomplete, currently based on Nathan's tutorial
    public static void buildVisualNeuralNet(GraphManager gm, String[][] data) throws FileNotFoundException {
        final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
        config.hideLabels();

        final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();

        int inputsCount;
        if (isTraining) {
            inputsCount = numAttributes + 1;
        } else {
            inputsCount = numAttributes;
        }

        prevA = Pipe.buildPipes(inputsCount, config);

        //TODO: refer to instance of our stage here
        inputStage.newInstance(gm, data, prevA);

        int nodesInLayerA = inputsCount;
        int nodesInLayerB = inputsCount;
        fromA = NeuralGraphBuilder.buildPipeLayer(gm, config, prevA, nodesInLayerA, factory);

        fromB = NeuralGraphBuilder.buildPipeLayer(gm, config, fromA, nodesInLayerB, factory);

        fromC = NeuralGraphBuilder.lastPipeLayer(gm, fromB, factory);

        //TODO: refer to instance of our output stage here
        outputStage.newInstance(gm, data, fromC, "");

    }

    public void initializeWeightMap() throws FileNotFoundException, IOException {
        //if we're in training mode, all weights stay at one
        if (isTraining) {
            //TODO how to assing weight to pipe after pulling from hashMap
            //TODO: pull weights from file here
            //TODO: add command line arguments fro weights file
            //TODO is overall repo structure ok? (.giingore, pom etc)
            BufferedReader b = new BufferedReader(new FileReader(weightsFN));
            String line;
            while ((line = b.readLine()) != null) {
                String k = line.split(" ")[0];
                Integer v = new Integer(line.split(" ")[1].replace("\n", ""));
                weightsMap.put(k, v);

            }
            b.close();
            for (int i = 0; i < prevA.length; i++) {
                prevA[i].weightsMap.get(prevA[i]);
            }
            for (int i = 0; i < fromA.length; i++) {
                for (int j = 0; j < fromA[i].length; j++) {
                    weightsMap.get(fromA[i][j]);
                }
            }
            for (int i = 0; i < fromB.length; i++) {
                for (int j = 0; j < fromB[i].length; j++) {
                    weightsMap.get(fromB[i][j]);
                }
            }
            for (int i = 0; i < fromC.length; i++) {
                weightsMap.get(fromC[i]);
            }
        } else {
            //discuss best init strategy
            for (int i = 0; i < prevA.length; i++) {
                weightsMap.put(prevA[i], new Float(1.0));
            }
            for (int i = 0; i < fromA.length; i++) {
                for (int j = 0; j < fromA[i].length; j++) {
                    weightsMap.put(fromA[i][j], new Float(1.0));
                }
            }
            for (int i = 0; i < fromB.length; i++) {
                for (int j = 0; j < fromB[i].length; j++) {
                    weightsMap.put(fromB[i][j], new Float(1.0));
                }
            }
            for (int i = 0; i < fromC.length; i++) {
                weightsMap.put(fromC[i], new Float(1.0));
            }
        }
    }
}
