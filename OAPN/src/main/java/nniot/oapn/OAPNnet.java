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
 * @author nick
 * @author bryson
 */
public class OAPNnet {

    static int numAttributes = 10;
    static int numTestRecords = 100;
    static int numTrainingRecords = 50;
    static String testDataFN = ""; //this file will not have classifications
    static String weightsInputFN = ""; //this file will have weights obtained via training
    static String weightsOutputFN = "";
    static String biasesInputFN = ""; //this file will have biases obtained via training
    static String biasesOutputFN = "";

    static String trainingDataFN = ""; // this file will already have classifications
    static Boolean isTraining = false;
    //This map is shared among all stages
    static HashMap<String, Float> weightsMap;
    static HashMap<String, Float> biasesMap;

    private static Appendable target;
    static Pipe<MessageSchemaDynamic>[] toFirstHiddenLayer;
    static Pipe<MessageSchemaDynamic>[][][] hiddenLayers;
    static Pipe<MessageSchemaDynamic>[] fromLastHiddenLayer;

    static int numHiddenLayers;
    static int numHiddenNodes;

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
            buildVisualNeuralNet(gm, trainingData, numHiddenLayers, numHiddenNodes);
        } else {
            buildVisualNeuralNet(gm, testingData, numHiddenLayers, numHiddenNodes);
        }

        gm.enableTelemetry(8089);

        StageScheduler.defaultScheduler(gm).startup();
    }

    public static void interpretCommandLineOptions(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-n":
                    numHiddenNodes = Integer.getInteger(args[i+1]);
                    i++;
                    break;
                case "-l":
                    numHiddenLayers = Integer.getInteger(args[i+1]);
                    i++;
                    break;
                case "-training":
                    isTraining = true;
                    break;
                case "-testing":
                    isTraining = false;
                    break;
                case "-t":
                    trainingDataFN = args[i+1];
                    i++;
                    break;
                case "-win":
                    weightsInputFN = args[i+1];
                    i++;
                    break;
                case "-wout":
                    weightsOutputFN = args[i+1];
                    i++;
                    break;
                case "-bin":
                    biasesInputFN = args[i+1];
                    i++;
                    break;
                case "-bout":
                    biasesOutputFN = args[i+1];
                    i++;
                    break;
                default:
                    System.out.println("usage: OAPNnet.java [-n <int>] "
                            + "[-l <int>] [-training | -testing] "
                            + "[-t <training_file.txt> | <testing_file.txt>] "
                            + "[-win <weights_input_file.txt>] "
                            + "[-wout <weights_output_file.txt>] "
                            + "[-bin <biases_input_file.txt>] "
                            + "[-bout <biases_output_file.txt>]\n");
                    System.out.println("-n\t\tSpecify number of nodes per hidden "
                            + "layer.");
                    System.out.println("-l\t\tSpecify number of hidden layers.");
                    System.out.println("-training\t\tIndicate if net is in "
                            + "training mode.");
                    System.out.println("-testing\t\tIndicate if net is in "
                            + "testing mode.");
                    System.out.println("-t\t\tSpecify name of file containing "
                            + "training/testing data.");
                    System.out.println("-win\t\tSpecify name of file to store "
                            + "weights after training. Defaults to "
                            + "./weights_input.txt.");
                    System.out.println("-wout\t\tSpecify name of file to store "
                            + "weights after training. Defaults to "
                            + "./weights_output.txt.");
                    System.out.println("-bin\t\tSpecify name of file to store "
                            + "biases after training. Defaults to "
                            + "./biases_input.txt.");
                    System.out.println("-bout\t\tSpecify name of file to store "
                            + "biases after training. Defaults to "
                            + "./biases_output.txt.");
                    return;
            }
        }
    }

    public static Float[][] readInData(Float[][] data, String fn) {
        String line;
        try {
            int i = 0;
            FileReader fileReader = new FileReader(fn);
            BufferedReader bufferedReader = new BufferedReader(fileReader);

            while ((line = bufferedReader.readLine()) != null) {
                String[] temp = line.split(",");
                for (int j = 0; j < temp.length; j++) {
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

    //Build OAPN Neural Net sized according to arguments numHiddenLayers and numHiddenNodes
    public static void buildVisualNeuralNet(GraphManager gm, Float[][] data, int numHiddenLayers, int numHiddenNodes) throws FileNotFoundException {
        final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
        config.hideLabels();

        final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();

        int inputsCount;
        if (isTraining) {
            inputsCount = numAttributes + 1;
        } else {
            inputsCount = numAttributes;
        }

        //Create initial pipe layer
        toFirstHiddenLayer = Pipe.buildPipes(inputsCount, config);
        inputStage.newInstance(gm, data, toFirstHiddenLayer);
        hiddenLayers[0] = NeuralGraphBuilder.buildPipeLayer(gm, config, toFirstHiddenLayer, numHiddenNodes, factory);

        //Create as many hidden layers as are specified by argument
        for (int i = 1; i < numHiddenLayers; i++) {
            hiddenLayers[i] = NeuralGraphBuilder.buildPipeLayer(gm, config, hiddenLayers[i - 1], numHiddenNodes, factory);
        }
        //Create final pipe layer
        fromLastHiddenLayer = NeuralGraphBuilder.lastPipeLayer(gm, hiddenLayers[hiddenLayers.length], factory);

        //Create instance of output stage
        outputStage.newInstance(gm, data, fromLastHiddenLayer, "");

    }
    
    
     /**
     * ***UNDER CONSTRUCTION***
     * Xavier initialization is used for weight initialization
     * It helps ensure that the weights aren't too small nor too large
     */
    public float XavierInitialization(float numNodes, float weight) {
        //numNodes * var(w) = 1
        //var(w) = 1/numNodes
        int mean = 0;
        numNodes = numHiddenNodes;
     
        //will need to find weights from Gaussian distribuation
        //with mean=0 and var=1/numNodes (double check?)
        
        return weight;
    }

    public void initializeWeightMap() throws FileNotFoundException, IOException {
        //if we're in training mode, all weights stay at one
        if (isTraining) {
            //TODO how to assign weight to pipe after pulling from hashMap
            //TODO: pull weights from file here
            //TODO: add command line arguments fro weights  and weights files
            //TODO is overall repo structure ok? (.giingore, pom etc)
            BufferedReader weightBR = new BufferedReader(new FileReader(weightsInputFN));
            BufferedReader biasBR = new BufferedReader(new FileReader(biasesInputFN));
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
            //Pull weights and biases for first layer from map
            for (int i = 0; i < toFirstHiddenLayer.length; i++) {
                this.weightsMap.get(toFirstHiddenLayer[i].toString());
                this.biasesMap.get(toFirstHiddenLayer[i].toString());
            }
            //Pull weights and biases for hidden layers from map
            for (int i = 0; i < hiddenLayers.length; i++) {
                for (int j = 0; j < hiddenLayers[i].length; j++) {
                    for (int k = 0; k < hiddenLayers[i].length; k++) {
                        weightsMap.get(hiddenLayers[i][j][k].toString());
                        biasesMap.get(hiddenLayers[i][j][k].toString());
                    }
                }
            }
            //Pull weights and biases for last hidden layer from map
            for (int i = 0; i < fromLastHiddenLayer.length; i++) {
                weightsMap.get(fromLastHiddenLayer[i].toString());
                biasesMap.get(fromLastHiddenLayer[i].toString());
            }
        } else {
            //Insert weights and biases for first layer into map
            for (int i = 0; i < toFirstHiddenLayer.length; i++) {
                weightsMap.put(toFirstHiddenLayer[i].toString(), new Float(1.0));
                biasesMap.put(toFirstHiddenLayer[i].toString(), new Float(0.0));
            }
            //Insert weights and biases for hidden layers into map
            for (int i = 0; i < hiddenLayers.length; i++) {
                for (int j = 0; j < hiddenLayers[i].length; j++) {
                    for (int k = 0; k < hiddenLayers[i][j].length; k++) {
                        weightsMap.put(hiddenLayers[i][j][k].toString(), new Float(1.0));
                        biasesMap.put(hiddenLayers[i][j][k].toString(), new Float(0.0));
                    }
                }
            }
            //Insert weights and biases for last hidden layer into map
            for (int i = 0; i < fromLastHiddenLayer.length; i++) {
                weightsMap.put(fromLastHiddenLayer[i].toString(), new Float(1.0));
                biasesMap.put(fromLastHiddenLayer[i].toString(), new Float(0.0));
            }
        }
    }
}
