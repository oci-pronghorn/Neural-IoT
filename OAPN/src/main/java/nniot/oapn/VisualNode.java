package nniot.oapn;

import java.util.Arrays;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import java.util.HashMap;

/**
 *
 * @author max
 */
public class VisualNode extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] input;
    private final Pipe<MessageSchemaDynamic>[] output;
    private float[] weights;

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic> input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = new Pipe[]{input};
        this.output = output;
        buildWeights();
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, input, output);
        this.input = input;
        this.output = output;
        buildWeights();
    }

    public VisualNode(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, Pipe<MessageSchemaDynamic> output) {
        super(gm, input, output);
        this.input = input;
        this.output = new Pipe[]{output};
        buildWeights();
    }

    private void buildWeights() {
        this.weights = new float[this.input.length];
        //loop pulls weights from singleton dictionary that uses pipes as keys
        for (int i = 0; i < input.length; i++) {
            weights[i] = OAPNnet.weightsMap.get(input[i]);
        }
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
                sum += (value * weights[i]);
            }

            //send this value to all the down stream nodes
            int j = output.length;
            float result = ReLu(sum);
            while (--j >= 0) {
                SchemalessPipe.writeFloat(output[j], result);
                SchemalessPipe.publishWrites(output[j]);
            }
        }
    }

    /**
     * Used to reduce the weighted sum calculated in run().
     */
    public float ReLu(float sum) {
        // Secondary option for rectifier function
        // return (float) Math.log(1 + Math.exp(sum)
        return Math.max(0, sum);
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

}
