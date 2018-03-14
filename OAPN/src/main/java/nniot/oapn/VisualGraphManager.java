/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nniot.oapn;


import com.ociweb.pronghorn.stage.scheduling.GraphManager;

/**
 *
 * @author max
 */
public class VisualGraphManager extends GraphManager{
    
    public static VisualNode[] allStages(GraphManager graphManager) {
        
        int count = 0;
        int s = graphManager.stageIdToStage.length;
        while (--s>=0) {
            PronghornStage stage = graphManager.stageIdToStage[s];             
            if (null!=stage) {
                count++;
            }
        }
        
        PronghornStage[] stages = new PronghornStage[count];
        s = graphManager.stageIdToStage.length;
        while (--s>=0) {
            PronghornStage stage = graphManager.stageIdToStage[s];             
            if (null != stage) {
                stages[--count] = stage;
            }
        }
        return stages;
    }
}
