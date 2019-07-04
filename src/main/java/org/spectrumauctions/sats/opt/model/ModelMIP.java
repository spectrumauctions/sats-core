package org.spectrumauctions.sats.opt.model;

import edu.harvard.econcs.jopt.solver.IMIP;
import edu.harvard.econcs.jopt.solver.ISolution;
import edu.harvard.econcs.jopt.solver.SolveParam;
import edu.harvard.econcs.jopt.solver.mip.MIP;
import edu.harvard.econcs.jopt.solver.mip.Variable;
import org.marketdesignresearch.mechlib.domain.Allocation;
import org.marketdesignresearch.mechlib.domain.bidder.Bidder;
import org.marketdesignresearch.mechlib.winnerdetermination.WinnerDetermination;

import java.util.Collection;

/**
 * @author Fabio Isler
 */
public abstract class ModelMIP extends WinnerDetermination {

    @Override
    protected abstract Allocation adaptMIPResult(ISolution mipResult); // Subclasses must implement this

    @Deprecated
    public Allocation calculateAllocation() {
        return getAllocation();
    }

    public abstract ModelMIP getMIPWithout(Bidder bidder);
    public abstract ModelMIP copyOf();

    private IMIP mip = new MIP();

    /**
     * This is mainly used for testing or if you need to access special SolveParams.
     * In most cases the setters from this class are sufficient.
     *
     * @return reference to the JOpt mip
     */
    public IMIP getMIP() {
        return mip;
    }

    /**
     * Defines the behaviour in case the solver hits the defined timeout.
     *
     * @param acceptSuboptimal true: accept a suboptimal solution at timeout; false: throw an exception at timeout
     */
    public void setAcceptSuboptimal(boolean acceptSuboptimal) {
        mip.setSolveParam(SolveParam.ACCEPT_SUBOPTIMAL, acceptSuboptimal);
    }

    /**
     * Defines the time limit for the solver.
     * What happens after the time limit is defined via {@link #setAcceptSuboptimal(boolean)}.
     *
     * @param timeLimit the time limit in seconds
     */
    public void setTimeLimit(double timeLimit) {
        mip.setSolveParam(SolveParam.TIME_LIMIT, timeLimit);
    }

    /**
     * ModelMIPs have to explicitly set variables of interest
     */
    @Override
    protected abstract Collection<Collection<Variable>> getVariablesOfInterest();


}
