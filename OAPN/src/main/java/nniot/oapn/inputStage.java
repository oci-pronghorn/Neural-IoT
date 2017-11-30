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
import java.util.HashMap;

public class inputStage extends PronghornStage {

    private final Pipe<MessageSchemaDynamic>[] output;
    
    public static inputStage newInstance(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output) {
        return new inputStage(gm, data, output);
    }

    public inputStage(GraphManager gm, String[][] data, Pipe<MessageSchemaDynamic>[] output) {
        super(gm, NONE, output);
        this.output = output;
    }

    public void run(){

        
    }
    
    public int roomForWrite() {
		int result = Integer.MAX_VALUE;
		int i = output.length;
		while (--i>=0) {
			result = Math.min(result, SchemalessPipe.roomRemaining(output[i]));
		}
		return result;
    }
}
