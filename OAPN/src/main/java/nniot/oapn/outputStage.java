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
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.logging.Level;
import java.util.logging.Logger;

public class outputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] output;

    private PrintWriter outputFileWriter;
    private File trainingFile;
    private final String[][] data;

    public static outputStage newInstance(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException {
        return new outputStage(gm, data, output, fname);

    }

    public outputStage(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) throws FileNotFoundException {
        super(gm, NONE, output);
        this.output = output;

        this.outputFileWriter = new PrintWriter(new File(fname));
        trainingFile = new File(fname.concat("OUTPUT"));
        this.data = data;

    }

    public void run() {
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
    public void writeOutput() throws FileNotFoundException {
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
        if (OAPNnet.isTraining) {
            outputFileWriter = new PrintWriter(trainingFile);
            for (Pipe<MessageSchemaDynamic> name : OAPNnet.weightsMap.keySet()) {
                String key = name.toString();
                String value = OAPNnet.weightsMap.get(name).toString();
                System.out.println(key + " " + value);
            }
        }
    }

    private String getCorrelatedOutput(String s[]) {
        // ask dr mayer and mr tippy
        return null;
    }
}
