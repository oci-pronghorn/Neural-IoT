/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
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
    private final String[][] data;

    public static outputStage newInstance(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException {
        return new outputStage(gm, data, output, fname);

    }

    public outputStage(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException, IOException {
        super(gm, NONE, output);
        this.output = output;

        this.outputFileWriter = new BufferedWriter(new FileWriter(new File(fname.concat("OUTPUT")),false));
        weightsFile = new File(fname.concat("OUTPUT-weights"));
        this.data = data;

    }

    public void run() throws IOException {
        try {
            writeOutput();
        } catch (FileNotFoundException ex) {
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
        for (String[] row : data) {
            String s = "";
            for (String item : row) {
                s += item;
                s += ", ";
            }
            s += getCorrelatedOutput(row);
            s += "\n";
            outputFileWriter.write(s);
        }
        outputFileWriter.close();
        if (OAPNnet.isTraining) {
            BufferedWriter out = new BufferedWriter(new FileWriter(weightsFile, false));
             for (Pipe<MessageSchemaDynamic> name : OAPNnet.weightsMap.keySet()) {
                 
                out.write(name.toString()+ " "+ OAPNnet.weightsMap.get(name)+"\n");

            }

            out.close();
                     
        }
    }

    private String getCorrelatedOutput(String s[]) {
        // ask dr mayer and mr tippy
        //TODO WHICH INPUT ROW GOES WITH WHICH OUTOUT, HASH EACH INPUT LINBE AND USE SINGLETON HASHMAP
        return null;
    }
}
