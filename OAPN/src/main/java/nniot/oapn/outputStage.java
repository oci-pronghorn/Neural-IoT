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
import java.util.HashMap;

public class outputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] output;
    private final File outputPath;
    
    public static outputStage newInstance(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) {
        return new outputStage(gm, data, output, fname);
    }

    public outputStage(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output, String fname) {
        super(gm, NONE, output);
        this.output = output;
        this.outputPath= new File("");
    }

    public void run(){

        
    }
    
    public int writeOutput() {
		int result = Integer.MAX_VALUE;
		int i = output.length;
		while (--i>=0) {
			result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
		}
		return result;
    }
}
