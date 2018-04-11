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
import java.util.logging.Level;
import java.util.logging.Logger;

public class OutputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] input;
    private int curDataExample;
    private static BufferedWriter outputFileWriter;
    //private File weightsFile;
    //private File biasesFile;

    private final Float[] data;

    public static void closeOutputFileWriter() throws IOException {
        outputFileWriter.close();
    }

    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, String fname) throws FileNotFoundException, IOException {
        super(gm, input, NONE);
        curDataExample = 0;
        this.input = input;
        if (outputFileWriter == null) {
            outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")), false));

        }
        //weightsFile = new File(fname.concat("OUTPUT-weights"));
        // biasesFile = new File(fname.concat("OUTPUT-biases"));
        this.data = null;
    }

    public static OutputStage newInstance(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, String fname) throws FileNotFoundException {
        OutputStage outputS = null;
        try {
            outputS = new OutputStage(gm, input, fname);
        } catch (IOException ex) {
            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputS;
    }

    @Override
    public void run() {
        if (!(data == null)) {
            try {
                writeOutput();
            } catch (FileNotFoundException ex) {
                Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
            } catch (IOException ex) {
                Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
            }
        }

    }

    /* public int writeOutput() {
     int result = Integer.MAX_VALUE;
     int i = output.length;
     while (--i>=0) {
     result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
     }
     return result;
     }*/
    public void writeOutput() throws FileNotFoundException, IOException {
        String curOutputLabel = translateToCorrectLabel(getMaxActivation());
        String op = "";
        for (int i = 0; i < data.length; i++) {
            op += data[i] + " ";
        }
        op += curOutputLabel;
        outputFileWriter.write(op);
        outputFileWriter.flush();
        curDataExample++;

    }

    /*
    Find the max activation values of the pipes coming into this stage in order
    to determine what class the NN thinks this example is.
     */
    private float getMaxActivation() {
        float maxActivation = Float.MIN_VALUE;

        while (availCount() == OAPNnet.numOutputNodes) {
            for (int i = 0; i < OAPNnet.numOutputNodes; i++) {
                float curr = SchemalessPipe.readFloat(input[i]);

                if (curr > maxActivation) {
                    maxActivation = curr;
                }
            }

        }

        return maxActivation;
    }

    /*
    Return an array of all activations that are coming into outputStage.
     */
    public float[] getAllOutputStageActivations() {
        float[] activations = new float[input.length];

        //for(int i = 0; i < input.length; i++) {
        int counter = 0;

        while (availCount() > 0) {
            activations[counter] = SchemalessPipe.readFloat(input[counter]);
            SchemalessPipe.releaseReads(input[counter]);
            counter++;
        }

        return activations;
    }

    private int availCount() {
        int avail = messagesToConsume();
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

    private String translateToCorrectLabel(float maxActivation) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
}
