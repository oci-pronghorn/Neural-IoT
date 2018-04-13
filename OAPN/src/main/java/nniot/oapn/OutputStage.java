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
    private float[] desired;
    //private File weightsFile;
    //private File biasesFile;

    private float[] data;

    public static void closeOutputFileWriter() throws IOException {
        outputFileWriter.close();
    }

    public OutputStage(GraphManager gm, Pipe<MessageSchemaDynamic>[] input, 
            String fname, float[] desired) throws FileNotFoundException, IOException {
        super(gm, input, NONE);
        curDataExample = 0;
        this.input = input;
        this.desired = desired;
        if (outputFileWriter == null) {
            outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")), false));

        }
        //weightsFile = new File(fname.concat("OUTPUT-weights"));
        // biasesFile = new File(fname.concat("OUTPUT-biases"));
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
                float curr = SchemalessPipe.readFloat(input[i]);
                SchemalessPipe.releaseReads(input[i]);
                data[i] = curr;
            }
        }
        
//        if (!(data == null)) {
//            try {
//                writeOutput();
//            } catch (FileNotFoundException ex) {
//                Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
//            } catch (IOException ex) {
//                Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
//            }
//        }
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
            op += data[i] + "";
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
    public float getMaxActivation() {
        float maxActivation = Float.MIN_VALUE;

        if (availCount() > 0) {
            for (int i = 0; i < input.length; i++) {
                float curr = SchemalessPipe.readFloat(input[i]);
                System.out.println("Max Activation: " + curr);
                SchemalessPipe.releaseReads(input[i]);

                if (curr > maxActivation) {
                    maxActivation = curr;
                }
            }

        }

        return maxActivation;
    }

    /**
     * Return an array of all activations that are coming into outputStage, and
     * release them.
     *
     * @return float[] an array of all outputs currently held in the stage
     */
    public float[] getAllOutputStageActivations() {
        float[] activations = null;

        if (availCount() > 0) {
            activations = new float[input.length];
            for (int i = 0; i < input.length; i++) {
                activations[i] = SchemalessPipe.readFloat(input[i]);
                //SchemalessPipe.releaseReads(input[i]);
            }
        }

        return activations;
    }

    private int availCount() {
        int avail = messagesToConsume();
        return avail;
    }

    private int messagesToConsume() {

        int results = Integer.MAX_VALUE;
        for (int i = 0; i < input.length; i++) {
            results = Math.min(results, SchemalessPipe.contentRemaining(input[i]));
        }

        return results;
    }

    private String translateToCorrectLabel(float maxActivation) {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }
    
    public float getCorrelatedOutput(int index) {
        return desired[index];
    }
}
