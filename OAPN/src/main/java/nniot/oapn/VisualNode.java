package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class VisualNode extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] input;
    private final Pipe<MessageSchemaDynamic>[] output;
    private float bias;
    private float[] weights;
    private float[] activations;
    private float activation; // activation value of this node
    private float z; // value of lastActivation * weight + bias, used in backpropogation

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = new Pipe[]{input};
        this.output = output;
        this.bias = 0;
        this.weights = new float[]{1.0f};
        this.activations = new float[1];
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = input;
        this.output = output;
        this.bias = 0;
        this.weights = new float[input.length];
        this.activations = new float[input.length];
        initializeWeights();
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
        super(gm, input, output);
        this.input = input;
        this.output = new Pipe[]{output};
        this.bias = 0;
        this.weights = new float[input.length];
        this.activations = new float[input.length];
        initializeWeights();
    }

    /**
     * Calculates the sum of the products of each incoming activation value and
     * the respective weight of the pipe sending it, rectifies the sum in the
     * ReLu() function, then passes the result downstream.
     */
    @Override
    public void run() {
        while (availCount() > 0) {
            int i = input.length;
            while (--i >= 0) {
                this.activations[i] = SchemalessPipe.readFloat(input[i]);
                SchemalessPipe.releaseReads(input[i]);
            }
            
            this.z = dot(activations, weights) + bias;
            
            //send this value to all the down stream nodes
            int j = output.length;
            this.activation = sigmoid(z);
            while (--j >= 0) {
                SchemalessPipe.writeFloat(output[j], this.activation);
                SchemalessPipe.publishWrites(output[j]);
            }
        }
    }

    /**
     * Used to rectify the weighted sum calculated in run().
     *
     * @param sum
     * @return
     */
    public float ReLu(float sum) {
        // Secondary option for rectifier function
        // return (float) Math.log(1 + Math.exp(sum))
        return Math.max(0, sum);
    }

    public float sigmoid(float z) {
        //return (float) Math.log(1 + Math.exp(z));
        return 1.0f / (1.0f + (float) Math.exp(-z));
    }

    private int availCount() {
        int avail = messagesToConsume();
        if (avail > 0) {
            avail = Math.min(avail, messagesOutputRoom());
        }
        return avail;
    }

    private int messagesToConsume() {
        int results = Integer.MAX_VALUE;
        int i = input.length;
        assert (i > 0);
        while (--i >= 0) {
            results = Math.min(results, SchemalessPipe.contentRemaining(input[i]));
        }
        return results;
    }

    private int messagesOutputRoom() {

        int results = Integer.MAX_VALUE;
        int i = output.length;
        assert (i > 0);
        while (--i >= 0) {
            results = Math.min(results, SchemalessPipe.roomRemaining(output[i]));
        }
        return results;
    }

    private void initializeWeights() {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = (float) Math.random();
        }
    }

    public static float dot(float[] a, float[] b) {
        if (a.length != b.length) {
            System.out.println("Dot product attempted between ArrayLists of different lengths.");
        }
        float retVal = 0.0f;

        for (int i = 0; i < a.length; i++) {
            retVal += a[i] * b[i];
        }

        return retVal;
    }

    public float getActivation() {
        return this.activation;
    }

    public float getBias() {
        return this.bias;
    }

    public float[] getWeights() {
        return this.weights;
    }

    public float getWeight(int index) {
        return this.weights[index];
    }

    public int getWeightsLength() {
        return this.weights.length;
    }

    public float getZ() {
        return this.z;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public void setWeights(float[] weights) {
        if (this.weights.length == weights.length) {
            for (int i = 0; i < weights.length; i++) {
                this.weights[i] = weights[i];
            }
        }
    }

    public void setWeight(int index, float weight) {
        this.weights[index] = weight;
    }
}
