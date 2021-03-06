/**
 * Copyright by Michael Weiss, weiss.michael@gmx.ch
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.spectrumauctions.sats.core.model.mrvm;

import com.google.common.base.Preconditions;

import org.marketdesignresearch.mechlib.core.allocationlimits.AllocationLimit;
import org.spectrumauctions.sats.core.bidlang.BiddingLanguage;
import org.spectrumauctions.sats.core.model.UnsupportedBiddingLanguageException;
import org.spectrumauctions.sats.core.model.World;
import org.spectrumauctions.sats.core.util.BigDecimalUtils;
import org.spectrumauctions.sats.core.util.random.RNGSupplier;
import org.spectrumauctions.sats.core.util.random.UniformDistributionRNG;
import org.spectrumauctions.sats.opt.model.mrvm.MRVM_MIP;

import java.math.BigDecimal;
import java.util.*;

/**
 * @author Michael Weiss
 *
 */
public final class MRVMRegionalBidder extends MRVMBidder {

    private static final long serialVersionUID = -3643980691138504665L;
    private final int homeId;
    transient MRVMRegionsMap.Region home;

    private final SortedMap<Integer, BigDecimal> distanceDiscounts;

    MRVMRegionalBidder(long id, long populationId, MRVMWorld world, MRVMRegionalBidderSetup setup,
                       UniformDistributionRNG rng, AllocationLimit limit) {
        super(id, populationId, world, setup, rng, limit);
        this.home = setup.drawHome(world, rng);
        this.homeId = home.getId();
        this.distanceDiscounts = new TreeMap<>(setup.drawDistanceDiscounts(world, home, rng));
        validateDistanceDiscounts(getWorld(), distanceDiscounts);
        store();
    }

    /**
     * Validates if a map with distanceDiscounts is valid for a given world, i.e., if it defines a valid discount for all possible distances
     * @param world the world
     * @param discounts a map of discounts
     * @throws NullPointerException if one of this methods argument is null or if a discount for a distance which is possible in the given is not defined
     * @throws IllegalArgumentException if one of the defined discounts for a feasible distance is negative.
     */
    private void validateDistanceDiscounts(MRVMWorld world, SortedMap<Integer, BigDecimal> discounts) {
        Preconditions.checkNotNull(world);
        Preconditions.checkNotNull(discounts);
        for (int i = 1; i < world.getRegionsMap().getLongestShortestPath(home); i++) {
            Preconditions.checkNotNull(discounts.get(i));
            Preconditions.checkArgument(discounts.get(i).compareTo(BigDecimal.ZERO) >= 0, "Discount must not be negative");
        }
    }


    public int getHomeId() {
        return homeId;
    }

    /**
     * {@inheritDoc}
     * If the two regions are not disconnected, the value is 0.
     * @param bundle Is not required for calculation of regional bidders gamma factors and will be ignored.
     */
    @Override
    public BigDecimal gammaFactor(MRVMRegionsMap.Region r, Set<MRVMLicense> bundle) {
        int distance = getWorld().getRegionsMap().getDistance(home, r);
        if (distance > distanceDiscounts.lastKey()) {
            //Not connected regions
            return BigDecimal.ZERO;
        }
        return distanceDiscounts.get(distance);
    }

    /**
     * {@inheritDoc}
     * @param bundle Is not required for calculation of regional bidders gamma factors and will be ignored.
     */
    @Override
    public Map<MRVMRegionsMap.Region, BigDecimal> gammaFactors(Set<MRVMLicense> bundle) {
        Map<MRVMRegionsMap.Region, BigDecimal> result = new HashMap<>();
        for (MRVMRegionsMap.Region region : getWorld().getRegionsMap().getRegions()) {
            // Note that repeatedly calculating distance is not expensive, as distance is cached in Map Instance
            int distance = getWorld().getRegionsMap().getDistance(home, region);
            BigDecimal discount = distanceDiscounts.getOrDefault(distance, BigDecimal.ZERO);
            result.put(region, discount);
        }
        return result;
    }

    @Override
    public MRVMRegionalBidder drawSimilarBidder(RNGSupplier rngSupplier) {
        return new MRVMRegionalBidder(getLongId(), getPopulation(), getWorld(), (MRVMRegionalBidderSetup) getSetup(), rngSupplier.getUniformDistributionRNG(), this.getAllocationLimit());
    }

    /* (non-Javadoc)
     * @see SATSBidder#getValueFunctionRepresentation(java.lang.Class, long)
     */
    @Override
    public <T extends BiddingLanguage> T getValueFunction(Class<T> type, RNGSupplier rngSupplier)
            throws UnsupportedBiddingLanguageException {
        return super.getValueFunction(type, rngSupplier);
    }

    /* (non-Javadoc)
     * @see SATSBidder#refreshReference(World)
     */
    @Override
    public void refreshReference(World world) {
        super.refreshReference(world);
        MRVMRegionsMap.Region homeCandidate = getWorld().getRegionsMap().getRegion(homeId);
        if (homeCandidate == null) {
            throw new IllegalArgumentException("The specified world does not have this bidders home region");
        } else {
            this.home = homeCandidate;
        }
    }

    @Override
    public int hashCode() {
        final int prime = 31;
        int result = super.hashCode();
        result = prime * result + ((distanceDiscounts == null) ? 0 : BigDecimalUtils.hashCodeIgnoringScale(distanceDiscounts));
        result = prime * result + homeId;
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (!super.equals(obj))
            return false;
        if (getClass() != obj.getClass())
            return false;
        MRVMRegionalBidder other = (MRVMRegionalBidder) obj;
        if (distanceDiscounts == null) {
            if (other.distanceDiscounts != null)
                return false;
        } else if (!BigDecimalUtils.equalIgnoreScaleOnValues(distanceDiscounts, other.distanceDiscounts))
            return false;
        if (homeId != other.homeId)
            return false;
        return true;
    }

	@Override
	protected void bidderTypeSpecificDemandQueryMIPAdjustments(MRVM_MIP mip) {
		// Do nothing
	}


}
