package nniot.oapn;

import com.ociweb.pronghorn.pipe.MessageSchemaDynamic;
import com.ociweb.pronghorn.pipe.Pipe;
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

    private BufferedWriter outputFileWriter;
    private File weightsFile;
    private File biasesFile;

    private final Float[][] data;

    public OutputStage(GraphManager gm, Float[][] data, Pipe<MessageSchemaDynamic>[] input, String fname) throws FileNotFoundException, IOException {
        super(gm, input, NONE);
        this.input = input;

        this.outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")), false));
        weightsFile = new File(fname.concat("OUTPUT-weights"));
        biasesFile = new File(fname.concat("OUTPUT-biases"));
        this.data = data;
    }
    
    public static OutputStage newInstance(GraphManager gm, Float[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException {
        OutputStage outputS = null;
        try {
            outputS = new OutputStage(gm, data, output, fname);
        } catch (IOException ex) {
            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputS;
    }

    @Override
    public void run() {
        try {
            writeOutput();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(OutputStage.class.getName()).log(Level.SEVERE, null, ex);
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
        for (Float[] row : data) {
            String s = "";
            for (Float item : row) {
                s += item.toString();
                s += ", ";
            }
            s += getCorrelatedOutput(row);
            s += "\n";
            outputFileWriter.write(s);
        }
        outputFileWriter.close();
        if (OAPNnet.isTraining) {
            BufferedWriter out = new BufferedWriter(new FileWriter(weightsFile, false));
            for (int i = 0; i < OAPNnet.nodesByLayer.size(); i++) {
                for (int j = 0; j < OAPNnet.nodesByLayer.get(i).length; j++) {
                    VisualNode node = OAPNnet.nodesByLayer.get(i)[j];
                    out.write(node.stageId);
                    for (int k = 0; k < node.input.length; k++) {
                        out.write(" " + node.input[k].toString() + "," + node.getWeight(k) + " ");
                    }
                    out.write("\n");
                }
            }

            out.close();
            out = new BufferedWriter(new FileWriter(biasesFile, false));
            for (int i = 0; i < OAPNnet.nodesByLayer.size(); i++) {
                for (int j = 0; j < OAPNnet.nodesByLayer.get(i).length; j++) {
                    VisualNode node = OAPNnet.nodesByLayer.get(i)[j];
                    out.write(node.stageId + " " + node.getBias() + "\n");
                }
            }

            out.close();

        }
    }

    private String getCorrelatedOutput(Float s[]) {
        // ask dr mayer and mr tippy
        //TODO: WHICH INPUT ROW GOES WITH WHICH OUTOUT, HASH EACH INPUT LINBE AND USE SINGLETON HASHMAP
        return null;
    }
}