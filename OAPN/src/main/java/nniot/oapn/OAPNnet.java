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

/**
 * @author Nick Kirkpatrick
 * @author Bryson Hunsaker
 * @author Max Spicer
 * @author Sandy Ly
 */
public class OAPNnet {

    static int numAttributes = 13;
    static int numTestRecords = 100;
    static int numTrainingRecords = 50;
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
    static Pipe<MessageSchemaDynamic>[][][] hiddenLayers;
    static Pipe<MessageSchemaDynamic>[] fromLastHiddenLayer;
    static ArrayList<VisualNode[]> nodesByLayer;

    static int numHiddenLayers = 2; // default = 2
    static int numHiddenNodes = 4; // default = 4

    public static void main(String[] args) throws FileNotFoundException {
        trainingData = new Float[numTrainingRecords][numAttributes + 1];
        testingData = new Float[numTestRecords][numAttributes];
        //String []   trainingAnswers = new String[numTrainingRecords];
        
        //System.out.println("Directory: " + System.getProperty("user.dir"));

        interpretCommandLineOptions(args);

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

        try {
            initializeWeightsAndBiases();
        } catch (FileNotFoundException e) {
            System.out.println("File " + e.toString() + " not found.");
        } catch (IOException e) {
            System.out.println("File " + e.toString() + "is inaccessible.");
        }

        System.out.println("Generating epochs...");
        generateEpochs(trainingData);
        //Create array containing each epoch's examples' outputs and the desired outputs
        //updateWeights() takes float[]
        for (int i = 0; i < epochsSet.length; i++) {
            for (int j = 0; j < epochsSet[i].length; j++) {
                updateWeights(epochsSet[i], 0.5f);
            }
        }
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

    //Build OAPN Neural Net sized according to arguments numHiddenLayers and numHiddenNodes
    //This is essentially forward propagation
    public static void buildVisualNeuralNet(GraphManager gm, Float[][] data, int numHiddenLayers, int numHiddenNodes) throws FileNotFoundException {
        final SchemalessFixedFieldPipeConfig config = new SchemalessFixedFieldPipeConfig(32);
        config.hideLabels();

        final StageFactory<MessageSchemaDynamic> factory = new VisualStageFactory();
        Set<PronghornStage> prevNodes = new HashSet<>();
        Set<PronghornStage> currNodes = new HashSet<>();
        Set<PronghornStage> temp = new HashSet<>();
        ArrayList<PronghornStage[]> stages = new ArrayList<>();
        nodesByLayer = new ArrayList<>();
        hiddenLayers = new Pipe[numHiddenLayers + 1][numHiddenNodes][numHiddenNodes];

        int inputsCount;
        if (isTraining) {
            inputsCount = numAttributes + 1;
        } else {
            inputsCount = numAttributes;
        }

        //Create initial pipe layer
        toFirstHiddenLayer = Pipe.buildPipes(inputsCount, config);

        //Create input layer nodes and add them to nodesByLayer ArrayList
        hiddenLayers[0] = NeuralGraphBuilder.buildPipeLayer(gm, config, toFirstHiddenLayer, numHiddenNodes, factory);

        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        stages.add(currNodes.toArray(new PronghornStage[0]));
        //System.out.println(stages.get(0).length);

        //Create data input stage, and add to allNodes so it is not counted as a data-holding stage
        inputStage.newInstance(gm, data, toFirstHiddenLayer);
        prevNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));

        /*
        //Create as many hidden layers as are specified by argument, add each layer to the nodesByLayer
        for (int i = 1; i < numHiddenLayers + 1; i++) {
            hiddenLayers[i] = NeuralGraphBuilder.buildPipeLayer(gm, config, hiddenLayers[i - 1], numHiddenNodes, factory);
            currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
            //System.out.println(currNodes);
            //System.out.println(prevNodes);
            temp.addAll(currNodes);
            currNodes.removeAll(prevNodes);
            prevNodes.addAll(temp);
            //System.out.println(prevNodes);
            stages.add(currNodes.toArray(new PronghornStage[0]));
            //System.out.println(stages.get(i).length);
        }*/
        
        //Create final pipe layer
        //fromLastHiddenLayer = NeuralGraphBuilder.lastPipeLayer(gm, hiddenLayers[numHiddenLayers - 1], factory);
        fromLastHiddenLayer = NeuralGraphBuilder.lastPipeLayer(gm, hiddenLayers[0], factory);

        
        //Create instance of output stage
        prevNodes.add(outputStage.newInstance(gm, data, fromLastHiddenLayer, ""));

        //Add output layer to nodesByLayer
        currNodes.addAll(Arrays.asList(GraphManager.allStages(gm)));
        currNodes.removeAll(prevNodes);
        //System.out.println(currNodes);
        stages.add(currNodes.toArray(new PronghornStage[0]));
        //System.out.println(stages.get(stages.size() - 1).length);

        for (int i = 0; i < stages.size(); i++) {
            VisualNode nodes[] = null;
            for (int j = 0; j < stages.get(i).length; j++) {
                nodes = new VisualNode[stages.get(i).length];
                //System.out.println("Stage at " + "(" + i + "," + j + ") = " + stages.get(i)[j].stageId);
                //System.out.println("Stage info: " + stages.get(i)[j].toString());
                VisualNode vn = (VisualNode) GraphManager.getStage(gm, stages.get(i)[j].stageId);
                //System.out.println("VN: " + vn.toString());
                nodes[j] = vn;
            }
            if (nodes != null) {
                nodesByLayer.add(i, nodes);
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

    /**
     * Initialize weights randomly for the neural net; values will be greater
     * than or equal to 0.00 and less than or equal to 1.0.
     *
     * @return float[]
     */
    public static float initializeWeight() {
        //float[] arr = new float[length];
        //for (int i = 0; i < length; i++) {
        //    arr[i] = (float) Math.random();
        //}
        //return arr;
        return (float) Math.random();
    }

    public static void initializeWeightsAndBiases() throws FileNotFoundException, IOException {
        //If we're in training mode, all weights stay at one
        if (isTraining) {
            for (int i = 0; i < nodesByLayer.size(); i++) {
                for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                    VisualNode node = nodesByLayer.get(i)[j];
                    for (int k = 0; k < node.input.length; k++) {
                        float weight = initializeWeight();
                        node.setWeight(k, weight);
                    }
                    node.setBias(new Float(0.0));
                }
            }
        } else {
            BufferedReader weightBR = new BufferedReader(new FileReader(weightsInputFN));
            BufferedReader biasBR = new BufferedReader(new FileReader(biasesInputFN));
            String line;
            
            //Traverse array of nodes by layer, if the key from the file matches the current node,
            //place that weight from file into a temporary array, which then goes into the correct node
            for (int i = 0; i < nodesByLayer.size(); i++) {
                for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                    float weightsForCurrNode[] = new float[nodesByLayer.get(i)[j].getWeights().length];
                    for (int k = 0; k < nodesByLayer.get(i)[j].getWeights().length; k++) {
                        if ((line = weightBR.readLine()) != null) {
                            String key = line.split(" ")[0];
                            Float v = new Float(line.split(" ")[1].replace("\n", ""));

                            if (nodesByLayer.get(i)[j].toString().equals(key)) {
                                weightsForCurrNode[k] = v;
                            }
                        }
                    }
                    nodesByLayer.get(i)[j].setWeights(weightsForCurrNode);
                }
            }
            
            //Traverse array of nodes by layer, if the key from the file matches the current node,
            //place that bias from file into the correct node
            for (int i = 0; i < nodesByLayer.size(); i++) {
                for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                    float biasForCurrNode =  0.0f;
                        if ((line = weightBR.readLine()) != null) {
                            String key = line.split(" ")[0];
                            Float v = new Float(line.split(" ")[1].replace("\n", ""));

                            if (nodesByLayer.get(i)[j].toString().equals(key)) {
                                biasForCurrNode = v;
                            }
                        }
                    nodesByLayer.get(i)[j].setBias(biasForCurrNode);
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
                for (int k = 0; k < nodesByLayer.get(i)[j].getWeights().length; k++) {
                    weightsBW.write(key + " " + nodesByLayer.get(i)[j].getWeights()[k] + "\n");
                }
                biasesBW.write(key + " " + nodesByLayer.get(i)[j].getBias() + "\n");
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
        ArrayList<ArrayList<Float>> newWeights = new ArrayList();
        ArrayList<ArrayList<Float>> weightDeltas = new ArrayList();
        ArrayList<ArrayList<Float>> biasDeltas = new ArrayList();
        float currWeight;
        float currBias;
        for (int i = 0; i < epoch.length; i++) {
            for (int j = 0; j < epoch[i].length; j++) {
                Object arrays[] = backpropagation(epoch[j][epoch.length - 1]);
                weightDeltas = (ArrayList<ArrayList<Float>>) arrays[0];
                biasDeltas = (ArrayList<ArrayList<Float>>) arrays[1];
            }
            for (int j = 0; j < nodesByLayer.size(); j++) {
                for (int k = 0; k < nodesByLayer.get(j).length; k++) {
                    for (int l = 0; l < nodesByLayer.get(j)[k].getWeightsLength(); l++) {
                        currWeight = nodesByLayer.get(j)[k].getWeight(l);
                        newWeights.get(k).set(l, currWeight + weightDeltas.get(k).get(l));
                    }
                    
                    float[] weights = new float[newWeights.get(k).size()];
                    currBias = nodesByLayer.get(j)[k].getBias();
                    
                    for (int l = 0; l < weights.length; l++) {
                        weights[l] = (float) newWeights.get(k).get(l);
                    }
                    
                    nodesByLayer.get(j)[k].setWeights(weights);
                    nodesByLayer.get(j)[k].setBias(currBias + biasDeltas.get(j).get(k));
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
        float a, b; // Z is a number that holds the weighted activation, e.g. z = (a * w) + b
        float[] w;
        ArrayList<ArrayList<Float>> zArray = new ArrayList(); // Holds all Z values
        ArrayList<ArrayList<Float>> activations = new ArrayList();
        ArrayList<ArrayList<Float>> layerWeights = new ArrayList();
        ArrayList<ArrayList<Float>> newBiases = new ArrayList();
        ArrayList<ArrayList<Float>> newWeights = new ArrayList();
        ArrayList<Float> activation = new ArrayList();
        ArrayList<Float> delta;
        ArrayList<Float> rp;
        // activations is an array of float[], first being output values of each node in last layer
        //  and then each sigmoid(z) return value appended
        for (VisualNode get : nodesByLayer.get(nodesByLayer.size() - 1)) {
            activation.add(get.getActivation());
        }
        activations.add(activation);
        
        for (int i = 0; i < nodesByLayer.size(); i++) {
            VisualNode[] nodes = nodesByLayer.get(i);
            ArrayList<Float> z;
            
            for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                a = nodes[j].getActivation();
                w = nodes[j].getWeights();
                b = nodes[j].getBias();
                z = new ArrayList();
                for (int k = 0; k < w.length; k++) {
                    z.add(a * w[k] + b);
                }
                
                zArray.add(z);
                activation = ReLuArray(z);
                activations.add(activation);
            }
        }
        
        delta = dot(costDerivative(activations.get(activations.size() - 1), desired), derivativeReLuArray(zArray.get(zArray.size() - 1)));
        newBiases.add(delta);
        newWeights.add(dot(delta, activations.get(activations.size() - 2))); //activations(size - 2) needs to be transposed (?)
        
        // TODO: fix matrix math; layerWeights is an nxn matrix and delta is a 1xn matrix, cannot be dot producted
        for (int i = 0; i < nodesByLayer.size(); i++) {
            ArrayList<Float> nodeWeights = new ArrayList();
            for (int j = 0; j < nodesByLayer.get(i).length; j++) {
                float arr[] = nodesByLayer.get(i)[j].getWeights();
                for (int k = 0; k < arr.length; k++) {
                    nodeWeights.add(arr[k]);
                }
            }
            layerWeights.add(nodeWeights);
        }
        
        for (int i = nodesByLayer.size() - 1; i > 0; i--) {
            ArrayList<Float> z = zArray.get(i);
            rp = derivativeReLuArray(z);
            for (int j = 0; j < layerWeights.size(); j++) {
                delta = dot(dot(layerWeights.get(j + 1), delta), rp);
                newBiases.add(delta);
                newWeights.add(dot(delta, activations.get(j - 1)));
            }
        }

        return new Object[]{newWeights, newBiases};
    }

    /**
     * Helper function used in backpropagation() to assign the delta of
     * each node.
     * @param arr
     * @return
     */
    public static ArrayList<Float> costDerivative(ArrayList<Float> arr, float desired) {
        ArrayList<Float> retArr = new ArrayList();
        for (float i : arr) {
            retArr.add(i - desired);
        }
        return retArr;
    }

    /**

     * Calls ReLu() for each element in array passed.
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
    
    public static ArrayList<Float> dot(ArrayList<Float> a, ArrayList<Float> b) {
        if (a.size() != b.size())
            System.out.println("Dot product attempted between ArrayLists of different lengths.");
        ArrayList<Float> retArr = new ArrayList();
        for (int i = 0; i < a.size(); i++) {
            retArr.add(i, a.get(i) * b.get(i));
        }
        return retArr;
    }
}
