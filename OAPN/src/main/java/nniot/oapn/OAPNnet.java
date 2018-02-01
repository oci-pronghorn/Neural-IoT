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
    static final String weightsFN = ""; //this file will have weights obtained via training 
    static final String biasesFN = ""; //this file will have biases obtained via training 
    
    
    static final String trainingDataFN = ""; // this file will already have classifications
    static Boolean isTraining = false;
    //This map is shared among all stages
    static HashMap<String, Float> weightsMap;
    static HashMap<String, Float> biasesMap;

    private static Appendable target;
    static Pipe<MessageSchemaDynamic>[][] fromA;
    static Pipe<MessageSchemaDynamic>[][] fromB;
    static Pipe<MessageSchemaDynamic>[] fromC;
    static Pipe<MessageSchemaDynamic>[] prevA;
    
    static int numLayers;
    static int numNodes;

    public static void main(String[] args) throws FileNotFoundException {
        Float[][] trainingData = new Float[numTestRecords][numAttributes + 1];
        Float[][] testingData = new Float[numTrainingRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];
        
        interpretCommandLineOptions(args);

        trainingData = readInData(trainingData, trainingDataFN);
        testingData = readInData(testingData, testDataFN);//todo why load both each time

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
    
    public static void interpretCommandLineOptions(String[] args) {
        for (String arg : args) {
            switch (arg) {
                case "help":
                    // Print help statement
                    break;
                case "-nodes":
                    
                    break;
                default: System.out.println("See 'OAPNnet help' for command line options.");
            }
        }
    }

    public static Float[][] readInData(Float[][] data, String fn) {
        String line = null;
        try {
            int i = 0;
            FileReader fileReader = new FileReader(fn);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while ((line = bufferedReader.readLine()) != null) {
                String[] temp = line.split(",");
                for(int j = 0; j < temp.length; j++){
                data[i][j] = Float.valueOf(temp[j]);                
                }
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
    public static void buildVisualNeuralNet(GraphManager gm, Float[][] data) throws FileNotFoundException {
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
        //TODO DO WE NEED TO ADD buildPipes for each
        //TODO Where should biases go
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
            //TODO how to assign weight to pipe after pulling from hashMap
            //TODO: pull weights from file here
            //TODO: add command line arguments fro weights  and weights files
            //TODO is overall repo structure ok? (.giingore, pom etc)
            BufferedReader weightBR = new BufferedReader(new FileReader(weightsFN));
            BufferedReader biasBR = new BufferedReader(new FileReader(biasesFN));
            String line;
            while ((line = weightBR.readLine()) != null) {
                String k = line.split(" ")[0];
                Float v = new Float(line.split(" ")[1].replace("\n", ""));
                weightsMap.put(k, v);

            }
            weightBR.close();
            while ((line = biasBR.readLine()) != null) {
                String k = line.split(" ")[0];
                Float v = new Float(line.split(" ")[1].replace("\n", ""));
                biasesMap.put(k, v);

            }
            biasBR.close();
            for (int i = 0; i < prevA.length; i++) {
                this.weightsMap.get(prevA[i]);
                this.biasesMap.get(prevA[i]);
            }
            for (int i = 0; i < fromA.length; i++) {
                for (int j = 0; j < fromA[i].length; j++) {
                    weightsMap.get(fromA[i][j]);
                    biasesMap.get(fromA[i][j]);
                }
            }
            for (int i = 0; i < fromB.length; i++) {
                for (int j = 0; j < fromB[i].length; j++) {
                    weightsMap.get(fromB[i][j]);
                    biasesMap.get(fromB[i][j]);
                }
            }
            for (int i = 0; i < fromC.length; i++) {
                weightsMap.get(fromC[i]);
                biasesMap.get(fromC[i]);
            }
        } else {
            //TODO discuss best init strategy for biases and weight
            for (int i = 0; i < prevA.length; i++) {
                weightsMap.put(prevA[i].toString(), new Float(1.0));
                biasesMap.put(prevA[i].toString(), new Float(0.0));
            }
            for (int i = 0; i < fromA.length; i++) {
                for (int j = 0; j < fromA[i].length; j++) {
                    weightsMap.put(fromA[i][j].toString(), new Float(1.0));
                    biasesMap.put(fromA[i][j].toString(), new Float(0.0));
                }
            }
            for (int i = 0; i < fromB.length; i++) {
                for (int j = 0; j < fromB[i].length; j++) {
                    weightsMap.put(fromB[i][j].toString(), new Float(1.0));
                    biasesMap.put(fromB[i][j].toString(), new Float(0.0));
                }
            }
            for (int i = 0; i < fromC.length; i++) {
                weightsMap.put(fromC[i].toString(), new Float(1.0));
                biasesMap.put(fromC[i].toString(), new Float(0.0));
            }
        }
    }
}
