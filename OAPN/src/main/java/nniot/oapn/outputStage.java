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
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class outputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] output;

    private BufferedWriter outputFileWriter;
    private File weightsFile;
    private File biasesFile;

    private final Float[][] data;

    public static outputStage newInstance(GraphManager gm, Float[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException {
        outputStage outputS = null;
        try {
            outputS = new outputStage(gm, data, output, fname);
        } catch (IOException ex) {
            Logger.getLogger(outputStage.class.getName()).log(Level.SEVERE, null, ex);
        }
        return outputS;
    }

    public outputStage(GraphManager gm, Float[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException, IOException {
        super(gm, NONE, output);
        this.output = output;

        this.outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")), false));
        weightsFile = new File(fname.concat("OUTPUT-weights"));
        biasesFile = new File(fname.concat("OUTPUT-biases"));
        this.data = data;

    }

    public void run() {
        try {
            writeOutput();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(outputStage.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(outputStage.class.getName()).log(Level.SEVERE, null, ex);
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
            for (String name : OAPNnet.weightsMap.keySet()) {

                out.write(name.toString() + " " + OAPNnet.weightsMap.get(name) + "\n");

            }

            out.close();
            out = new BufferedWriter(new FileWriter(biasesFile, false));
            for (String name : OAPNnet.biasesMap.keySet()) {

                out.write(name.toString() + " " + OAPNnet.biasesMap.get(name) + "\n");

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
