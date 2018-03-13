package nniot.oapn;

import com.ociweb.pronghorn.neural.NeuralGraphBuilder;
import com.ociweb.pronghorn.neural.StageFactory;
import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessFixedFieldPipeConfig;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import com.ociweb.pronghorn.stage.scheduling.StageScheduler;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Set;
import java.util.HashSet;

/**
 * @author Nick Kirkpatrick
 * @author Bryson Hunsaker
 * @author Max Spicer
 * @author Sandy Ly
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
    static HashMap<String, Float> weightsMap;   // associated with pipes
    static HashMap<String, Float> biasesMap;    // associated with nodes
    static HashMap<String, Float> errorsMap;    // associated with nodes
    
    static Float[][] trainingData;
    static Float[][] testingData;
    
    static Float[][][] epochsSet;

    private static Appendable target;
    static Pipe<MessageSchemaDynamic>[] toFirstHiddenLayer;
    static Pipe<MessageSchemaDynamic>[][][] hiddenLayers;
    static Pipe<MessageSchemaDynamic>[] fromLastHiddenLayer;
    static ArrayList<VisualNode[]> nodesByLayer;

    static int numHiddenLayers;
    static int numHiddenNodes;

    public static void main(String[] args) throws FileNotFoundException {
        trainingData = new Float[numTestRecords][numAttributes + 1];
        testingData = new Float[numTrainingRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];

        interpretCommandLineOptions(args);

        target = null;

        GraphManager gm = new GraphManager();
        GraphManager.addDefaultNota(gm, GraphManager.SCHEDULE_RATE, 20_000);
        if (isTraining) {
            trainingData = readInData(trainingData, trainingDataFN);
            
            buildVisualNeuralNet(gm, trainingData, numHiddenLayers, numHiddenNodes);
        } else {
            testingData = readInData(testingData, testDataFN);
            buildVisualNeuralNet(gm, testingData, numHiddenLayers, numHiddenNodes);
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
        int[] epochIndices = new int[numEpochs];
        
        for (Float[] row : inputData) {
            int epochSetIndex = -1;
            
            while (epochSetIndex < 0 || epochsSet[epochSetIndex][epochIndices[epochSetIndex]].length > epochSize) {
                epochSetIndex = (int) Math.floor(Math.random() * numEpochs);
            }
            
            epochsSet[epochSetIndex][epochIndices[epochSetIndex]] = row;
            epochIndices[epochSetIndex] = epochIndices[epochSetIndex] + 1;
        }
        
        return epochsSet;
    }

    //Build OAPN Neural Net sized according to arguments numHiddenLayers and numHiddenNodes
    //This is essentially forward propagation
    public static void buildVisualNeuralNet(GraphManager gm, Float[][] data, int numHiddenLayers, int numHiddenNodes) throws FileNotFoundException {
        final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
        config.hideLabels();

        final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
        Set<VisualNode> allNodes = new HashSet<>();
        Set<VisualNode> currNodes = new HashSet<>();
        Set<VisualNode> temp = new HashSet<>();
        nodesByLayer = new ArrayList<>();

        int inputsCount;
        if (isTraining) {
            inputsCount = numAttributes + 1;
        } else {
            inputsCount = numAttributes;
        }

        //Create initial pipe layer
        toFirstHiddenLayer = Pipe.buildPipes(inputsCount, config);
        inputStage.newInstance(gm, data, toFirstHiddenLayer);
        
        //Create input layer nodes and add them to nodesByLayer ArrayList
        hiddenLayers[0] = NeuralGraphBuilder.buildPipeLayer(gm, config, toFirstHiddenLayer, numHiddenNodes, factory);
        allNodes.addAll(Arrays.asList((VisualNode[]) GraphManager.allStages(gm)));
        currNodes.addAll(Arrays.asList((VisualNode[]) GraphManager.allStages(gm)));
        nodesByLayer.add(currNodes.toArray(new VisualNode[1]));

        //Create as many hidden layers as are specified by argument, add each layer to the nodesByLayer
        for (int i = 1; i < numHiddenLayers; i++) {
            hiddenLayers[i] = NeuralGraphBuilder.buildPipeLayer(gm, config, hiddenLayers[i - 1], numHiddenNodes, factory);
            currNodes.addAll(Arrays.asList((VisualNode[]) GraphManager.allStages(gm)));
            temp = currNodes;
            currNodes.removeAll(allNodes);
            allNodes = temp;
            nodesByLayer.add(currNodes.toArray(new VisualNode[1]));
        }
        //Create final pipe layer
        fromLastHiddenLayer = NeuralGraphBuilder.lastPipeLayer(gm, hiddenLayers[hiddenLayers.length], factory);

        //Create instance of output stage
        outputStage.newInstance(gm, data, fromLastHiddenLayer, "");
        
        //Add output layer to nodesByLayer
        currNodes.addAll(Arrays.asList((VisualNode[]) GraphManager.allStages(gm)));
        currNodes.removeAll(allNodes);
        nodesByLayer.add(currNodes.toArray(new VisualNode[1]));
    }
    
    /**
     * Initialize initial weights randomly for forward propagation; values 
     * will be greater than or equal to 0.00 and less than or equal to 1.0.
     * @return float
     */
    public float initializeWeight() {
        return (float) Math.random();
    }
    
    public void initializeWeightMap() throws FileNotFoundException, IOException {
        //If we're in training mode, all weights stay at one
        if (isTraining) {
            //Insert weights and biases for first layer into map
            for (int i = 0; i < toFirstHiddenLayer.length; i++) {
                weightsMap.put(toFirstHiddenLayer[i].toString(), initializeWeight());
                biasesMap.put(toFirstHiddenLayer[i].toString(), new Float(0.0));
            }
            //Insert weights and biases for hidden layers into map
            for (int i = 0; i < hiddenLayers.length; i++) {
                for (int j = 0; j < hiddenLayers[i].length; j++) {
                    for (int k = 0; k < hiddenLayers[i][j].length; k++) {
                        weightsMap.put(hiddenLayers[i][j][k].toString(), initializeWeight());
                        //Insert weights and biases for last hidden layer into map
                        biasesMap.put(hiddenLayers[i][j][k].toString(), new Float(0.0));
                    }
                }
            }
            //Insert weights and biases for last hidden layer into map
            for (int i = 0; i < fromLastHiddenLayer.length; i++) {
                weightsMap.put(fromLastHiddenLayer[i].toString(), initializeWeight());
                biasesMap.put(fromLastHiddenLayer[i].toString(), new Float(0.0));
            }
        } else {
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
        }
    }
    
    /**
     * Function to save current weights and biases of net. Used if net is 
     * terminated, but will be resumed at a later time.
     * @param weightsOutputFile
     * @param biasesOutputFile
     * @throws IOException
     */
    public void saveWeightMap(String weightsOutputFile, String biasesOutputFile)
            throws IOException {
        BufferedWriter weightsBW = new BufferedWriter(new FileWriter(
                weightsOutputFile));
        BufferedWriter biasesBW = new BufferedWriter(new FileWriter(
                biasesOutputFile));
        
        for (int i = 0; i < toFirstHiddenLayer.length; i++) {
            String key = toFirstHiddenLayer[i].toString();
            weightsBW.write(key + " " + weightsMap.get(key) + "\n");
            biasesBW.write(key + " " + biasesMap.get(key) + "\n");
        }
        
        for (int i = 0; i < hiddenLayers.length; i++) {
            for (int j = 0; j < hiddenLayers[i].length; j++) {
                for (int k = 0; k < hiddenLayers[i][j].length; k++) {
                    String key = hiddenLayers[i][j][k].toString();
                    weightsBW.write(key + " " + weightsMap.get(key) + "\n");
                    biasesBW.write(key + " " + biasesMap.get(key) + "\n");
                }
            }
        }
        
        for (int i = 0; i < fromLastHiddenLayer.length; i++) {
            String key = fromLastHiddenLayer[i].toString();
            weightsBW.write(key + " " + weightsMap.get(key) + "\n");
            biasesBW.write(key + " " + biasesMap.get(key) + "\n");
        }
        
        weightsBW.close();
        biasesBW.close();
    }
    
    /**
     * Function to update the weight of each connection based on the node's last
     * activation and its error. Calls backpropagation() to find new weights.
     * @param epoch
     * @param desired
     * @param learningRate
     */
    public void updateWeights(float[] epoch, float desired, float learningRate) {
        HashMap<String, Float> newWeights = new HashMap();
        HashMap<String, Float> newBiases = new HashMap();
        HashMap<String, Float> weightDeltas;
        HashMap<String, Float> biasDeltas;
        
        for(int j = 0; j < epoch.length; j++) {
            Object arrays[] = backpropagation(desired);
            weightDeltas = (HashMap<String, Float>) arrays[0];
            biasDeltas = (HashMap<String, Float>) arrays[1];
            
            Iterator itW = weightsMap.entrySet().iterator();
            Iterator itB = biasesMap.entrySet().iterator();
            String wKey, bKey;
            for (int k = 0; k < weightsMap.size(); k++) {
                wKey = (String) ((Map.Entry)itW.next()).getKey();
                bKey = (String) ((Map.Entry)itB.next()).getKey();
                
                newWeights.put(wKey, weightsMap.get(wKey) + weightDeltas.get(wKey));
                newBiases.put(bKey, biasesMap.get(bKey) + biasDeltas.get(bKey));
            }
        }
        
        Iterator itW = weightsMap.entrySet().iterator();
        Iterator itB = biasesMap.entrySet().iterator();
        String wKey, bKey;
        for (int k = 0; k < weightsMap.size(); k++) {
            wKey = (String) ((Map.Entry)itW.next()).getKey();
            bKey = (String) ((Map.Entry)itB.next()).getKey();
            
            weightsMap.put(wKey, weightsMap.get(wKey) - (learningRate/epoch.length) * newWeights.get(wKey));
            biasesMap.put(bKey, biasesMap.get(bKey) - (learningRate/epoch.length) * newWeights.get(bKey));
        }
    }
    
    /**
     * A major step of neural network training is backwards propagation of error
     * and activation values. This function finds the error of each node in each
     * layer and stores that value in the node to be used in updateWeights().
     * @param desired
     * @return 
     */
    public Object[] backpropagation(float desired) {
        // Find a way to grab all activations from neural network at any given point, store in HashMap
        HashMap<String, Float> activations = new HashMap();
        HashMap<String, Float> newWeights = new HashMap();
        HashMap<String, Float> newBiases = new HashMap();
        
        Iterator itA = activations.entrySet().iterator();
        Iterator itW = weightsMap.entrySet().iterator();
        Iterator itB = biasesMap.entrySet().iterator();
        
        float z, a = 0, w, b, dr; // Z is a number that holds the weighted activation, e.g. z = (a * w) + b
        ArrayList<Float> zArray = new ArrayList(); // Holds all Z values
        
        while (itW.hasNext() && itA.hasNext()) {
            a = (float) ((Map.Entry) itA.next()).getValue();
            w = (float) ((Map.Entry) itW.next()).getValue();
            if (itB.hasNext())
                b = (float) ((Map.Entry) itB.next()).getValue();
            else
                b = 0.0f;
                
            z =  a * w + b;
            zArray.add(z);
        }
        
        float delta = (a - desired) * derivativeReLu(zArray.get(zArray.size() - 1));
        itW = weightsMap.entrySet().iterator();
        itB = biasesMap.entrySet().iterator();
        VisualNode lastLayer[] = nodesByLayer.get(nodesByLayer.size() - 1);
        
        while (itB.hasNext()) {
            Map.Entry pair = (Map.Entry) itB.next();
            for (int i = 0; i < lastLayer.length; i++) {
                if (((String) pair.getKey()).equals(lastLayer[i].toString()))
                    pair.setValue(delta);
            }
        }
        
        while (itW.hasNext()) {
            Map.Entry pair = (Map.Entry) itW.next();
            for (int i = 0; i < lastLayer.length; i++) {
                if (((String) pair.getKey()).equals(lastLayer[i].toString())) // Need to find each node in the last layer and give it the appropriate delta
                    pair.setValue(delta * lastLayer[i].getActivation()); // Need to find the output value of the node found above and assign a new value
            }
        }
        
        for (int i = numHiddenLayers; i > 0 ; i--) {
            z = zArray.get(zArray.size() - i);
            dr = derivativeReLu(z);
            for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                delta = weightsMap.get(nodesByLayer.get(i + 1)[j].toString()) * delta * dr;
                newBiases.put(nodesByLayer.get(i)[j].toString(), delta);
                newWeights.put(nodesByLayer.get(i - 1)[j].toString(), delta * nodesByLayer.get(i)[j].getActivation());
            }
        }
                
        return new Object[]{newWeights, newBiases};
    }
    
//    /**
//     * Calculates the cost function of the network used in back propagation.
//     * Returns the updated weights and biases to be used in the network for the
//     * next epoch.
//     * @param layerIndex
//     * @param trainingDataIndex
//     * @return 
//     */
//    public float[] calculateCost(int layerIndex, int trainingDataIndex) {
//        // TODO: Finish cost function
//        // Note: currently written recursively, may be changed to more
//        //       typical iterative style
//        
//        if (layerIndex == 0)
//            // return value of the input nodes' activations
//            return epochsSet[0][trainingDataIndex];
//        
//        float currentLayer[];
//        
//        if (layerIndex == numHiddenLayers + 1)
//            currentLayer = fromLastHiddenLayer;
//        else
//            currentLayer = hiddenLayers[layerIndex];
//        
//        // need set of desired output values for each piece of training data
//        // z_j = sum(weight_i * activation_i) + bias
//        //  z is weighted sum of a layer
//        // cost_0 = sum((activation_i - desiredOutput_i)^2)
//        // delCost_0 / delAct = 2(activation - desiredOutput)
//        // delAct / delZ = derivative of sigmoid
//        // delZ / delWeight = Act of last layer
//        
//        float desiredOutput = trainingData[trainingDataIndex][numAttributes + 1];
//        float weightedSum = 0.0f; // z
//        for (int i = 0; i < hiddenLayers.length + 2; i++) {
//            for (int j = 0; j < hiddenLayers[i].length; j++) {
//                for (int k = 0; k < hiddenLayers[i][j].length; k++) {
//                    // weightedSum += weight of each pipe * activation of connected node in current layer
//                    weightedSum += weightsMap.get(hiddenLayers[i][j][k].toString()) * hiddenLayers[i][j][k].value;
//                }
//            }
//        }
//        
//        float costGradientBiases[biasesMap.size()];
//        float costGradientWeights[weightsMap.size()];
//        
//        for (int i = 0; i < currentLayer.length; i++) {
//            // delC / delA = 2(current layer node's result - desiredOutput)
//            float delCdelA = 2 * (currentLayer[i].result - desiredOutput);
//
//            // delA / delZ = derivative of rectifying function(z), z = weighted sum
//            float delAdelZ = derivativeReLu(weightedSum);
//
//            // delZ / delW = activation of node of last layer
//            float delZdelW = calculateCost(layerIndex - 1, trainingDataIndex - 1);
//            
//            float delZdelB = 1;
//
//            // delC / delW = product of each other partial derivative
//            float delCdelW = delCdelA * delAdelZ * delZdelW;
//            
//            float delCdelB = 
//            costGradient[i] = delCdelW;
//        }
//
//        return costGradient;
//    }
    
    /**
     * Helper function used in backpropagation() to assign the delta of
     * each node.
     * @param node
     * @return
     */
    public float calculateDelta(VisualNode node) {
        return errorsMap.get(node.toString()) * calculateDerivative(node.getActivation());
    }
    
    /**
     * Helper function used in back propagation for calculating error of hidden
     * layers.
     * @param value
     * @return
     */
    public float calculateDerivative(float value) {
        return value * (1.0f - value);
    }
    
    public float derivativeReLu(float sum) {
        if (sum > 0) {
            return 1.0f;
        } else {
            return 0.0f;
        }
    }
}
