package nniot.oapn;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import com.ociweb.pronghorn.stage.PronghornStage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;
import java.util.Map;
import java.util.HashMap;

/**
 * @author Nick Kirkpatrick
 * @author Bryson Hunsaker
 * @author Max Spicer
 * @author Sandy Ly
 */
public class OAPNnet {

    static String testDataFN = ""; //this file will not have classifications
    static String weightsInputFN = ""; //this file will have weights obtained via training
    static String weightsOutputFN = "";
    static String biasesInputFN = ""; //this file will have biases obtained via training
    static String biasesOutputFN = "";

    static String trainingDataFN = "./wineTraining.data"; // this file will already have classifications
    static Boolean isTraining = true;

    static Float[][] trainingData;
    static Float[][] testingData;

    static Float[][][] epochsSet; // 3d-array of each epoch, containing examples, which contain their attributes

    static Pipe<MessageSchemaDynamic>[] toFirstHiddenLayer;
    static Pipe<MessageSchemaDynamic>[][][] layers;
    static Pipe<MessageSchemaDynamic>[] fromLastHiddenLayer;
    static ArrayList<PronghornStage[]> nodesByLayer;
    static InputStage input;
    static OutputStage output;

    static int numAttributes = 13;
    static int numTestRecords = 100;
    static int numTrainingRecords = 50;
    static int numHiddenLayers = 2; // default = 2
    static int numHiddenNodes = 4; // default = 4
    static int numOutputNodes = 3;

    public static void main(String[] args) throws FileNotFoundException {
        trainingData = new Float[numTrainingRecords][numAttributes + 1];
        testingData = new Float[numTestRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];

        //System.out.println("Directory: " + System.getProperty("user.dir"));
        interpretCommandLineOptions(args);

        GraphManager gm = new GraphManager();
        GraphManager.combineCommonEdges = false;
        GraphManager.addDefaultNota(gm, GraphManager.SCHEDULE_RATE, 20_000);
        if (isTraining) {
            trainingData = readInData(trainingData, trainingDataFN);
            preprocessData();
            buildVisualNeuralNet(gm, numHiddenLayers, numHiddenNodes);

            try {
                initializeWeightsAndBiases();
            } catch (FileNotFoundException e) {
                System.out.println("File " + e.toString() + " not found.");
            } catch (IOException e) {
                System.out.println("File " + e.toString() + "is inaccessible.");
            }

            System.out.println("Generating epochs...");
            generateEpochs(trainingData);
            System.out.println("Finished generating epochs...");
            //Create array containing each epoch's examples' outputs and the desired outputs
            //updateWeights() takes float[]

            for (int i = 0; i < epochsSet.length; i++) {
                //printNeuralNet();
                updateWeights(epochsSet[i], 0.5f);
            }

            gm.enableTelemetry(8089);
            StageScheduler.defaultScheduler(gm).startup();

        } else {
            testingData = readInData(testingData, testDataFN);
            //buildVisualNeuralNet(gm, testingData, numHiddenLayers, numHiddenNodes);

            try {
                initializeWeightsAndBiases();
            } catch (FileNotFoundException e) {
                System.out.println("File " + e.toString() + " not found.");
            } catch (IOException e) {
                System.out.println("File " + e.toString() + "is inaccessible.");
            }

            // TODO: Logic for testing new data sets goes here
        }

        gm.enableTelemetry(8089);
        StageScheduler.defaultScheduler(gm).startup();
    }

    public static void interpretCommandLineOptions(String[] args) {
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-n":
                    numHiddenNodes = Integer.getInteger(args[i + 1]);
                    i++;
                    break;
                case "-l":
                    numHiddenLayers = Integer.getInteger(args[i + 1]);
                    i++;
                    break;
                case "-training":
                    isTraining = true;
                    break;
                case "-testing":
                    isTraining = false;
                    break;
                case "-t":
                    trainingDataFN = args[i + 1];
                    i++;
                    break;
                case "-win":
                    weightsInputFN = args[i + 1];
                    i++;
                    break;
                case "-wout":
                    weightsOutputFN = args[i + 1];
                    i++;
                    break;
                case "-bin":
                    biasesInputFN = args[i + 1];
                    i++;
                    break;
                case "-bout":
                    biasesOutputFN = args[i + 1];
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
                    System.out.println("-n\t\tSpecify number of nodes per "
                            + "hidden layer.");
                    System.out.println("-l\t\tSpecify number of hidden "
                            + "layers.");
                    System.out.println("-training\t\tIndicate if net is in "
                            + "training mode.");
                    System.out.println("-testing\t\tIndicate if net is in "
                            + "testing mode.");
                    System.out.println("-t\t\tSpecify name of file containing "
                            + "training/testing data.");
                    System.out.println("-win\t\tSpecify name of file to store "
                            + "weights if already trained. Defaults to "
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

    public static void preprocessData() {
        int numClasses = 0;
        ArrayList<Float> classes = new ArrayList<>();
        for (int i = 0; i < trainingData.length; i++) {
            if (!classes.contains(trainingData[i][0])) {
                classes.add(trainingData[i][0]);
                numClasses++;
            }
        }

        numOutputNodes = numClasses;
        numAttributes = trainingData[0].length - 1;
    }

    //Build OAPN Neural Net sized according to arguments numHiddenLayers and numHiddenNodes
    //This is essentially forward propagation
    public static void buildVisualNeuralNet(GraphManager gm, int numHiddenLayers, int numHiddenNodes) throws FileNotFoundException {
        final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
        final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
        Set<PronghornStage> prevNodes = new HashSet<>();
        Set<PronghornStage> currNodes = new HashSet<>();
        Set<PronghornStage> temp = new HashSet<>();
        ArrayList<PronghornStage[]> stages = new ArrayList<>();
        //toFirstHiddenLayer = new Pipe[numAttributes][numHiddenNodes];
        nodesByLayer = new ArrayList<>();
        layers = new Pipe[numHiddenLayers + 2][numHiddenNodes][numHiddenNodes];

        config.hideLabels();

        // Returns array of Pipe objects to be used in building input layer
        toFirstHiddenLayer = Pipe.buildPipes(numAttributes, config);

        //Create input stage that splits data into its attributes and passes it to first VisualNode layer
        input = InputStage.newInstance(gm, toFirstHiddenLayer);

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        prevNodes.addAll(currNodes);

        layers[0] = NeuralGraphBuilder.buildPipeLayer(gm, config, toFirstHiddenLayer, numHiddenNodes, factory);

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        prevNodes.addAll(currNodes);

        //Create as many hidden layers as are specified by argument, add each layer to the stages array
        for (int i = 1; i < numHiddenLayers; i++) {
            layers[i] = NeuralGraphBuilder.buildPipeLayer(gm, config, layers[i - 1], numHiddenNodes, factory);
            currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
            temp.addAll(currNodes);
            currNodes.removeAll(prevNodes);
            stages.add(currNodes.toArray(new PronghornStage[0]));
            prevNodes.addAll(temp);
        }

        layers[numHiddenLayers] = NeuralGraphBuilder.buildPipeLayer(gm, config, layers[numHiddenLayers - 1], numOutputNodes, factory);

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        prevNodes.addAll(currNodes);

        fromLastHiddenLayer = NeuralGraphBuilder.lastPipeLayer(gm, layers[numHiddenLayers], factory);

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        prevNodes.addAll(currNodes);

        //Create data consumer layer
        output = OutputStage.newInstance(gm, fromLastHiddenLayer, "");

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        prevNodes.addAll(currNodes);

        for (int i = 0; i < stages.size(); i++) {
            if (stages.get(i).length > 0) {
                PronghornStage nodes[];
                if (i == 0) {
                    nodes = new InputStage[stages.get(i).length];
                } else if (i == stages.size() - 1) {
                    nodes = new OutputStage[stages.get(i).length];
                } else {
                    nodes = new VisualNode[stages.get(i).length];
                }

                for (int j = 0; j < stages.get(i).length; j++) {
                    System.out.println("Stage at " + "(" + i + "," + j + ") = " + stages.get(i)[j].stageId);
                    PronghornStage stage = (PronghornStage) GraphManager.getStage(gm, stages.get(i)[j].stageId);
                    nodes[j] = stage;
                }

                if (nodes != null) {
                    nodesByLayer.add(nodes);
                }
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
            System.out.println("Unable to open file: " + fn);
        } catch (IOException ex) {
            System.out.println("IOException while reading file: " + fn);
        }
        return data;
    }

    public static Float[][][] generateEpochs(Float[][] inputData) {
        int epochSize = (int) Math.ceil(inputData.length / 10.0f);
        int numEpochs = (int) Math.ceil(inputData.length / epochSize);
        epochsSet = new Float[numEpochs][epochSize][numAttributes + 1];
        int[] epochSizeCounter = new int[numEpochs];
        int epochSetIndex;

        for (Float[] row : inputData) {
            do {
                epochSetIndex = (int) Math.floor(Math.random() * numEpochs);
                if (epochSizeCounter[epochSetIndex] < epochSize) {
                    epochsSet[epochSetIndex][epochSizeCounter[epochSetIndex]] = row;
                    epochSizeCounter[epochSetIndex] = epochSizeCounter[epochSetIndex] + 1;
                    row = null;
                }

            } while (row != null);
        }

        return epochsSet;
    }

    public static void initializeWeightsAndBiases() throws FileNotFoundException, IOException {
        if (!isTraining) {
            System.out.println("Loading weights and biases from files...");
            BufferedReader weightBR = new BufferedReader(new FileReader(weightsInputFN));
            BufferedReader biasBR = new BufferedReader(new FileReader(biasesInputFN));
            String line;

            //Traverse array of nodes by layer, if the key from the file matches the current node,
            //place that weight from file into a temporary array, which then goes into the correct node
            for (int i = 0; i < nodesByLayer.size(); i++) {
                for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                    float weightsForCurrNode[] = new float[((VisualNode) nodesByLayer.get(i)[j]).getWeights().length];
                    for (int k = 0; k < ((VisualNode) nodesByLayer.get(i)[j]).getWeights().length; k++) {
                        if ((line = weightBR.readLine()) != null) {
                            String key = line.split(" ")[0];
                            Float v = new Float(line.split(" ")[1].replace("\n", ""));

                            if (nodesByLayer.get(i)[j].toString().equals(key)) {
                                weightsForCurrNode[k] = v;
                            }
                        }
                    }
                    ((VisualNode) nodesByLayer.get(i)[j]).setWeights(weightsForCurrNode);
                }
            }

            //Traverse array of nodes by layer, if the key from the file matches the current node,
            //place that bias from file into the correct node
            for (int i = 0; i < nodesByLayer.size(); i++) {
                for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                    float biasForCurrNode = 0.0f;
                    if ((line = weightBR.readLine()) != null) {
                        String key = line.split(" ")[0];
                        Float v = new Float(line.split(" ")[1].replace("\n", ""));

                        if (nodesByLayer.get(i)[j].toString().equals(key)) {
                            biasForCurrNode = v;
                        }
                    }
                    ((VisualNode) nodesByLayer.get(i)[j]).setBias(biasForCurrNode);
                }
            }
            weightBR.close();
            biasBR.close();
        }
    }

    /**
     * Function to save current weights and biases of net. Used if net is
     * terminated, but will be resumed at a later time.
     *
     * @param weightsOutputFile
     * @param biasesOutputFile
     * @throws IOException
     */
    public static void writeWeightsBiases(String weightsOutputFile, String biasesOutputFile)
            throws IOException {
        BufferedWriter weightsBW = new BufferedWriter(new FileWriter(
                weightsOutputFile));
        BufferedWriter biasesBW = new BufferedWriter(new FileWriter(
                biasesOutputFile));

        for (int i = 0; i < nodesByLayer.size(); i++) {
            for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                String key = nodesByLayer.get(i)[j].toString();
                for (int k = 0; k < ((VisualNode) nodesByLayer.get(i)[j]).getWeights().length; k++) {
                    weightsBW.write(key + " " + ((VisualNode) nodesByLayer.get(i)[j]).getWeights()[k] + "\n");
                }
                biasesBW.write(key + " " + ((VisualNode) nodesByLayer.get(i)[j]).getBias() + "\n");
            }
        }

        weightsBW.close();
        biasesBW.close();
    }

    /**
     * Function to update the weight of each connection based on the node's last
     * activation and its error. Calls backpropagation() to find new weights.
     *
     * @param epoch
     * @param learningRate
     */
    public static void updateWeights(Float[][] epoch, float learningRate) {
        Map<Integer, ArrayList<Float>> newWeights;
        Map<Integer, Float> newBiases;
        float[] exampleData;
        for (int i = 0; i < epoch.length; i++) {
            exampleData = new float[epoch[i].length];
            for (int j = 1; j < epoch[i].length; j++) {
                exampleData[j] = epoch[i][j];
            }
            input.giveInputData(exampleData);
            while(output.getMaxActivation() == Float.MIN_VALUE) {
                ;
            }
            Object arrays[] = backpropagation(epoch[i][0]);
            newWeights = (HashMap<Integer, ArrayList<Float>>) arrays[0];
            newBiases = (HashMap<Integer, Float>) arrays[1];
            for (int j = 2; j < nodesByLayer.size(); j++) {
                if (j != nodesByLayer.size() - 1) {
                    for (int k = 0; k < nodesByLayer.get(j).length; k++) {
                        VisualNode node = (VisualNode) nodesByLayer.get(j)[k];
                        for (int l = 0; l < node.getWeightsLength(); l++) {
                            node.setWeight(l, node.getWeight(l) - (learningRate
                                    / epoch[i].length) * newWeights.get(node.stageId).get(l));
                        }

                        node.setBias(node.getBias() - (learningRate / epoch[i].length) * newBiases.get(node.stageId));
                    }
                }
            }
        }
    }

    /**
     * A major step of neural network training is backwards propagation of error
     * and activation values. This function finds the error of each node in each
     * layer and stores that value in the node to be used in updateWeights().
     *
     * @param desired
     * @return
     */
    public static Object[] backpropagation(float desired) {
        Map<Integer, ArrayList<Float>> newWeights = new HashMap();
        Map<Integer, Float> newBiases = new HashMap();
        ArrayList<ArrayList<Float>> layerWeightsMatrix;
        ArrayList<Float> layerWeightsArray;
        ArrayList<Float> zArray = new ArrayList();
        ArrayList<Float> layerActivations = new ArrayList();
        ArrayList<Float> costDerivative = new ArrayList();
        ArrayList<Float> delta;
        ArrayList<Float> nodeWeights;
        float newWeight;
        float[] w;

        // For each output value, add that value to the activation array
        float[] outputs = ((OutputStage) nodesByLayer.get(nodesByLayer.size() - 1)[0]).getAllOutputStageActivations();
        for (int i = 0; i < outputs.length; i++) {
            zArray.add(outputs[i]);
        }

        PronghornStage[] nodes = nodesByLayer.get(nodesByLayer.size() - 2);
        for (int i = 0; i < nodes.length; i++) {
            costDerivative.add(((VisualNode) nodes[i]).getActivation() - desired);
        }

        nodes = nodesByLayer.get(nodesByLayer.size() - 3);
        for (int i = 0; i < nodes.length; i++) {
            layerActivations.add(((VisualNode) nodes[i]).getActivation());
        }

        // Delta value for output layer
        delta = hadamardProduct(costDerivative, ReLuArray(zArray));

        // Assign new biases and weights for output layer
        for (int j = 0; j < delta.size(); j++) {
            nodeWeights = new ArrayList();
            for (int k = 0; k < layerActivations.size(); k++) {
                newWeight = delta.get(j) * layerActivations.get(k);
                nodeWeights.add(newWeight);
            }
            newBiases.put(nodesByLayer.get(nodesByLayer.size() - 2)[j].stageId, delta.get(j));
            newWeights.put(nodesByLayer.get(nodesByLayer.size() - 2)[j].stageId, nodeWeights);
        }

        // Start i at size - 3 because we skip the OutputStage layer (size - 1), 
        // and the last VisualNodes layer is done above
        for (int i = nodesByLayer.size() - 3; i > 1; i--) {
            zArray = new ArrayList();
            layerActivations = new ArrayList();
            layerWeightsMatrix = new ArrayList();

            // Get the activations from the next layer
            nodes = nodesByLayer.get(i - 1);
            for (int j = 0; j < nodes.length; j++) {
                layerActivations.add(((VisualNode) nodes[j]).getActivation());
            }

            // Get the weights between last layer and current
            nodes = nodesByLayer.get(i + 1);
            for (int j = 0; j < nodes.length; j++) {
                layerWeightsArray = new ArrayList();

                w = ((VisualNode) nodes[j]).getWeights();
                for (int k = 0; k < w.length; k++) {
                    layerWeightsArray.add(w[k]);
                }
                layerWeightsMatrix.add(layerWeightsArray);
            }

            nodes = nodesByLayer.get(i);
            for (int j = 0; j < nodes.length; j++) {
                zArray.add(((VisualNode) nodes[j]).getWeightedSum());
            }

            // Delta value for i-th layer
            delta = hadamardProduct(matMult(transpose(layerWeightsMatrix), delta), ReLuArray(zArray));

            for (int j = 0; j < delta.size(); j++) {
                nodeWeights = new ArrayList();
                for (int k = 0; k < layerActivations.size(); k++) {
                    newWeight = delta.get(j) * layerActivations.get(k);
                    nodeWeights.add(newWeight);
                }
                newBiases.put(nodesByLayer.get(i)[j].stageId, delta.get(j));
                newWeights.put(nodesByLayer.get(i)[j].stageId, nodeWeights);
            }
        }

        return new Object[]{newWeights, newBiases};
    }

    /**
     * Helper function used in backpropogation to transpose activations array
     */
    private static ArrayList<ArrayList<Float>> transpose(ArrayList<ArrayList<Float>> arr) {
        ArrayList<ArrayList<Float>> transposed;

        if (arr.get(0).size() == arr.size()) {
            transposed = new ArrayList(arr.size());
            for (int i = 0; i < arr.size(); i++) {
                transposed.add(i, new ArrayList(arr.size()));
            }
            for (int i = 0; i < arr.size(); i++) {
                for (int j = 0; j < arr.get(i).size(); j++) {
                    transposed.get(j).add(i, arr.get(i).get(j));
                }
            }
        } else {
            transposed = new ArrayList(arr.get(0).size());
            for (int i = 0; i < arr.get(0).size(); i++) {
                transposed.add(i, new ArrayList(arr.size()));
            }

            for (int i = 0; i < arr.size(); i++) {
                for (int j = 0; j < arr.get(i).size(); j++) {
                    transposed.get(j).add(i, arr.get(i).get(j));
                }
            }
        }

        return transposed;
    }

    /**
     * Calls ReLu() for each element in array passed.
     *
     * @param arr is an ArrayList of floats
     * @return ArrayList of floats
     */
    public static ArrayList<Float> ReLuArray(ArrayList<Float> arr) {
        ArrayList<Float> retArr = new ArrayList();
        for (float i : arr) {
            retArr.add(ReLu(i));
        }
        return retArr;
    }

    public static float ReLu(float i) {
        // Secondary option for rectifier function
        // return (float) Math.log(1 + Math.exp(i))
        return Math.max(0, i);
    }

    public static ArrayList<Float> derivativeReLuArray(ArrayList<Float> arr) {
        ArrayList<Float> retArr = new ArrayList();
        for (float i : arr) {
            retArr.add(derivativeReLu(i));
        }
        return retArr;
    }

    public static float derivativeReLu(float sum) {
        if (sum > 0) {
            return 1.0f;
        } else {
            return 0.0f;
        }
    }

    public static float dot(ArrayList<Float> a, ArrayList<Float> b) {
        if (a.size() != b.size()) {
            System.out.println("Dot product attempted between ArrayLists of different lengths.");
        }
        float retVal = 0.0f;

        for (int i = 0; i < a.size(); i++) {
            retVal += a.get(i) * b.get(i);
        }

        return retVal;
    }

    public static ArrayList<Float> matMult(ArrayList<ArrayList<Float>> arr1, ArrayList<Float> arr2) {
        if (arr1.get(0).size() != arr2.size()) {
            System.out.println("Attempted to multiply incompatible matrices.");
            return null;
        }

        ArrayList<Float> retArr = new ArrayList();
        float element = 0.0f;

        for (int i = 0; i < arr1.size(); i++) {
            for (int j = 0; j < arr1.get(i).size(); j++) {
                element += arr1.get(i).get(j) * arr2.get(j);
            }
            retArr.add(element);
        }
        return retArr;
    }

    public static ArrayList<Float> vectMult(ArrayList<Float> arr1, ArrayList<Float> arr2) {
        if (arr1.size() != arr2.size()) {
            System.out.println("Attempted to multiply incompatible vectors.");
            return null;
        }

        ArrayList<Float> retArr = new ArrayList();
        for (int i = 0; i < arr1.size(); i++) {
            retArr.add(arr1.get(i) * arr2.get(i));
        }
        return retArr;
    }

    public static ArrayList<Float> vectAdd(ArrayList<Float> arr1, ArrayList<Float> arr2) {
        if (arr1.size() != arr2.size()) {
            System.out.println("Attempted to add incompatible vectors.");
            return null;
        }

        ArrayList<Float> retArr = new ArrayList();

        for (int i = 0; i < arr1.size(); i++) {
            retArr.add(arr1.get(i) + arr2.get(i));
        }
        return retArr;
    }

    public static ArrayList<Float> hadamardProduct(ArrayList<Float> arr1, ArrayList<Float> arr2) {
        if (arr1.size() != arr2.size()) {
            System.out.println("Attempted to calculate Hadamard Product of incompatible matrices.");
            return null;
        }

        ArrayList<Float> retArr = new ArrayList();
        for (int i = 0; i < arr1.size(); i++) {
            retArr.add(arr1.get(i) * arr2.get(i));
        }
        return retArr;
    }

    public static float[] listToArray(ArrayList<Float> arr) {
        float[] retArr = new float[arr.size()];
        for (int i = 0; i < arr.size(); i++) {
            retArr[i] = arr.get(i);
        }
        return retArr;
    }

    public static ArrayList<String> printNeuralNet() {
        ArrayList<String> strings = new ArrayList();
        String s;
        for (int i = 0; i < nodesByLayer.size(); i++) {
            s = "";
            for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                PronghornStage stage = nodesByLayer.get(i)[j];
                s += "(" + stage.stageId + ":";

                if (i == 0) {
                    s += " InputStage)";
                } else if (i == nodesByLayer.size() - 1) {
                    s += " OutputStage)";
                } else {
                    VisualNode node = (VisualNode) stage;
                    s += " VisualNode";
                    for (int k = 0; k < node.getWeightsLength(); k++) {
                        s += " w" + k + "=" + node.getWeight(k);
                    }
                    s += " b=" + node.getBias() + ")";
                }
            }
            strings.add(s);
        }

        System.out.println(strings);
        return strings;
    }
}
