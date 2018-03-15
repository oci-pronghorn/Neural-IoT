package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;

public class VisualNode extends PronghornStage {

    public final Pipe<MessageSchemaDynamic>[] input;
    public final Pipe<MessageSchemaDynamic>[] output;
    private float bias;
    private float[] weights;
    private float result; //activation value

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = new Pipe[]{input};
        this.output = output;
        this.bias = 0;
        this.weights = new float[] { 0.0f };
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = input;
        this.output = output;
        this.bias = 0;
        this.weights = new float[input.length];
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
        super(gm, input, output);
        this.input = input;
        this.output = new Pipe[]{output};
        this.bias = 0;
        this.weights = new float[input.length];
    }

    /**
     * Calculates the sum of the products of each incoming activation value and
     * the respective weight of the pipe sending it, rectifies the sum in the
     * ReLu() function, then passes the result downstream.
     */
    @Override
    public void run() {
        while (availCount() > 0) {

            float sum = 0;
            int i = input.length;
            while (--i >= 0) {
                float value = SchemalessPipe.readFloat(input[i]);
                SchemalessPipe.releaseReads(input[i]);
                sum += (value * weights[i]) + bias;
            }

            //send this value to all the down stream nodes
            int j = output.length;
            this.result = ReLu(sum);
            while (--j >= 0) {
                SchemalessPipe.writeFloat(output[j], this.result);
                SchemalessPipe.publishWrites(output[j]);
            }
        }
    }

    /**
     * Used to reduce the weighted sum calculated in run().
     * @param sum
     * @return
     */
    public float ReLu(float sum) {
        // Secondary option for rectifier function
        // return (float) Math.log(1 + Math.exp(sum))
        return Math.max(0, sum);
    }

    public float derivativeReLu(float sum) {
        if (sum > 0) {
            return 1.0f;
        } else {
            return 0.0f;
        }
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
    
        public float getActivation() {
        return result;
    }
    
    public float getBias() {
        return bias;
    }
    
    public float[] getWeights() {
        return weights;
    }
    
    public int getWeightsLength() {
        return weights.length;
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
