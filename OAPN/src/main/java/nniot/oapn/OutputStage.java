package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
import com.ociweb.pronghorn.pipe.SchemalessPipe;
import com.ociweb.pronghorn.stage.PronghornStage;
import static com.ociweb.pronghorn.stage.PronghornStage.NONE;
import com.ociweb.pronghorn.stage.scheduling.GraphManager;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Queue;
import java.util.logging.Level;
import java.util.logging.Logger;

public class OutputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] input;
    private static BufferedWriter outputFileWriter;
    private final float[] desired;
    private float[] data;
    public Queue<float[]> buffer = new ArrayDeque();

    public static void closeOutputFileWriter() throws IOException {
        outputFileWriter.close();
    }

    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input,
            String fname, float[] desired) throws FileNotFoundException, IOException {
        super(gm, input, NONE);
        this.input = input;
        this.desired = desired;
        if (outputFileWriter == null) {
            outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")), false));
        }
        this.data = null;
    }

    public static OutputStage newInstance(GraphManager gm,
            Pipe<MessageSchemaDynamic>[] input, String fname, float[] desired) throws FileNotFoundException {
        OutputStage outputS = null;
        try {
            outputS = new OutputStage(gm, input, fname, desired);
        } catch (IOException ex) {
            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputS;
    }

    @Override
    public void run() {
        if (availCount() > 0) {
            data = new float[input.length];
            for (int i = 0; i < input.length; i++) {
                data[i] = SchemalessPipe.readFloat(input[i]);
                SchemalessPipe.releaseReads(input[i]);
            }
            buffer.add(data);
        }
    }

    /**
     * Find the max activation values of the pipes coming into this stage in order
     * to determine what class the NN thinks this example is.
     * @return the highest value of this stage's data[] field
     */
    public float getMaxActivation() {
        float maxActivation = Float.MIN_VALUE;
        if (data != null) {
            for (int i = 0; i < data.length; i++) {
                if (data[i] > maxActivation) {
                    maxActivation = data[i];
                }
            }
        }

        return maxActivation;
    }
    
    public float[] getData() {
        return data;
    }

    private int availCount() {
        int avail = messagesToConsume();
        return avail;
    }

    /**
     * Returns the minimum number of messages waiting in this stage's input pipes.
     * @return 
     */
    private int messagesToConsume() {
        int results = Integer.MAX_VALUE;
        for (int i = 0; i < input.length; i++) {
            //  / input[i].sizeOfSlabRing
            results = Math.min(results, SchemalessPipe.contentRemaining(input[i]));
        }

        return results;
    }

    public float getCorrelatedOutput(int index) {
        return desired[index];
    }
}
