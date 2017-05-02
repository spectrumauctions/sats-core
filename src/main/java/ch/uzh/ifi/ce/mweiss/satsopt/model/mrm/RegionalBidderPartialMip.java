/**
 * Copyright by Michael Weiss, weiss.michael@gmx.ch
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package ch.uzh.ifi.ce.mweiss.satsopt.model.mrm;

import java.math.BigDecimal;

import ch.uzh.ifi.ce.mweiss.specval.model.mrm.MRMRegionalBidder;
import ch.uzh.ifi.ce.mweiss.specval.model.mrm.MRMRegionsMap.Region;
import edu.harvard.econcs.jopt.solver.mip.CompareType;
import edu.harvard.econcs.jopt.solver.mip.Constraint;
import edu.harvard.econcs.jopt.solver.mip.MIP;
import edu.harvard.econcs.jopt.solver.mip.Variable;

/**
 * @author Michael Weiss
 *
 */
public class RegionalBidderPartialMip extends BidderPartialMIP{

    private final MRMRegionalBidder bidder;
    /**
     * @param bidder
     * @param worldMip
     */
    public RegionalBidderPartialMip(MRMRegionalBidder bidder, double scalingFactor, WorldPartialMip worldMip) {
        super(bidder, scalingFactor, worldMip);
        this.bidder = bidder;
    }

    public Constraint constrainValue(){
        Constraint constraint = new Constraint(CompareType.EQ, 0);
        Variable biddersValue = worldPartialMip.getValueVariable(bidder);
        constraint.addTerm(-1,biddersValue);
        for(Region region : bidder.getWorld().getRegionsMap().getRegions()){
            BigDecimal gammaFactor = bidder.gammaFactor(region, null); //The discount factor in [0,1]
            double constant = gammaFactor.doubleValue();
            Variable regionalOmega = getOmegaVariable(region);
            constraint.addTerm(constant, regionalOmega);
        }
        return constraint;
    }

    /* (non-Javadoc)
     * @see ch.uzh.ifi.ce.mweiss.satsopt.imip.PartialMIP#appendToMip(edu.harvard.econcs.jopt.solver.mip.MIP)
     */
    @Override
    public void appendToMip(MIP mip) {
        super.appendToMip(mip);
        mip.add(constrainValue()); 
    }
    
}
