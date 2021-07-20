package org.spectrumauctions.sats.core.model.lsvm;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Lists;
import edu.harvard.econcs.jopt.solver.mip.*;
import lombok.Getter;
import lombok.Setter;
import org.marketdesignresearch.mechlib.core.Allocation;
import org.marketdesignresearch.mechlib.core.Bundle;
import org.marketdesignresearch.mechlib.core.allocationlimits.AllocationLimit;
import org.marketdesignresearch.mechlib.core.price.Prices;
import org.marketdesignresearch.mechlib.instrumentation.MipInstrumentation;
import org.spectrumauctions.sats.core.bidlang.BiddingLanguage;
import org.spectrumauctions.sats.core.bidlang.xor.DecreasingSizeOrderedXOR;
import org.spectrumauctions.sats.core.bidlang.xor.IncreasingSizeOrderedXOR;
import org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR;
import org.spectrumauctions.sats.core.model.SATSBidder;
import org.spectrumauctions.sats.core.model.UnsupportedBiddingLanguageException;
import org.spectrumauctions.sats.core.model.World;
import org.spectrumauctions.sats.core.util.random.RNGSupplier;
import org.spectrumauctions.sats.opt.model.lsvm.LSVMStandardMIP;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

/**
 * @author Fabio Isler
 */
public final class LSVMBidder extends SATSBidder {

    private static final long serialVersionUID = -1774118565772856391L;
    private final int LSVM_A;
    private final int LSVM_B;
    private final List<LSVMLicense> proximity;
    private final HashMap<Long, BigDecimal> values;
    private transient LSVMWorld world;
    private final String description;
    @Getter
    @Setter
    private AllocationLimit allocationLimit = AllocationLimit.NO;

    LSVMBidder(LSVMBidderSetup setup, LSVMWorld world, long currentId, long population, RNGSupplier rngSupplier) {
        super(setup, population, currentId, world.getId());
        this.world = world;
        LSVMLicense favorite = setup.drawFavorite(rngSupplier, world);
        if (setup.getProximitySize() == -1) { // It's a national bidder
            this.proximity = world.getGrid().getLicenses();
        } else { // It's a regional bidder
            this.proximity = world.getGrid().getProximity(favorite, setup.getProximitySize());
        }
        this.values = setup.drawValues(rngSupplier, this);
        this.LSVM_A = setup.getLsvmA();
        this.LSVM_B = setup.getLsvmB();
        this.description = setup.getSetupName() + " which has its headquarter in " + favorite.getName() +
                ", thus interested in licenses "
                + this.proximity.stream().map(LSVMLicense::getName).collect(Collectors.joining(", "))
                + ".";
        store();
    }

    public ImmutableSet<LSVMLicense> getProximity() {
        return ImmutableSet.copyOf(proximity);
    }

    @Override
    public BigDecimal calculateValue(Bundle bundle) {
        double value = 0;
        Set<LSVMLicense> licences;
        if(world.isLegacyLSVM())
        	licences = bundle.getBundleEntries().stream().map(be -> (LSVMLicense) be.getGood()).collect(Collectors.toSet());
        else 
        	licences = bundle.getBundleEntries().stream().map(be -> (LSVMLicense) be.getGood()).filter(l -> this.getProximity().contains(l)).collect(Collectors.toSet());
        Set<Set<LSVMLicense>> subpackages = world.getGrid().getMaximallyConnectedSubpackages(licences);
        for (Set<LSVMLicense> subset : subpackages) {
            double factor = calculateFactor(subset.size());
            value += factor * sumOfItemValues(subset);
        }
        return BigDecimal.valueOf(value);
    }

    private double sumOfItemValues(Set<LSVMLicense> subset) {
        double value = 0;
        for (LSVMLicense license : subset) {
            if (this.values.containsKey(license.getLongId())) {
                value += this.values.get(license.getLongId()).doubleValue();
            }
        }
        return value;
    }

    @Override
    public <T extends BiddingLanguage> T getValueFunction(Class<T> clazz, RNGSupplier rngSupplier) throws UnsupportedBiddingLanguageException {
        if (clazz.isAssignableFrom(SizeBasedUniqueRandomXOR.class)) {
            return clazz.cast(
                    new SizeBasedUniqueRandomXOR(world.getLicenses(), rngSupplier, this));
        } else if (clazz.isAssignableFrom(IncreasingSizeOrderedXOR.class)) {
            return clazz.cast(
                    new IncreasingSizeOrderedXOR(world.getLicenses(), this));
        } else if (clazz.isAssignableFrom(DecreasingSizeOrderedXOR.class)) {
            return clazz.cast(
                    new DecreasingSizeOrderedXOR(world.getLicenses(), this));
        } else {
            throw new UnsupportedBiddingLanguageException();
        }
    }

    /**
     * @see SATSBidder#getWorld()
     */
    @Override
    public LSVMWorld getWorld() {
        return world;
    }

    @Override
    public void refreshReference(World world) {
        Preconditions.checkArgument(world.getId() == getWorldId());
        if (world instanceof LSVMWorld) {
            this.world = (LSVMWorld) world;
        } else {
            throw new IllegalArgumentException("World is not of correct type");
        }
    }

    @Override
    public LSVMBidder drawSimilarBidder(RNGSupplier rngSupplier) {
        return new LSVMBidder((LSVMBidderSetup) getSetup(), getWorld(), getLongId(), getPopulation(), rngSupplier);
    }

    public Map<Long, BigDecimal> getBaseValues() {
        return Collections.unmodifiableMap(values);
    }

    /**
     * This factor is used to calculate the bonus for having adjacent items.
     * <p>
     * Note: The current interpretation of Scheffel et al. (2010) is such that a bidder has no positive marginal value
     * for any licenses he is not interested in. (I.e. adding or removing such a license from a bundle does not change
     * the value of the bundle). There are mainly two arguments for this interpretation:
     * 1.  Scheffel et al. (2010) states that the bidding space of the regional bidder consists of 63 - 2047 packages (bundles).
     *     As a regional bidder is interested in 6 to 11 licenses this makes clear that the bidding space of a regional bidder
     *     does not include any licences he is not interested in
     * 2.  This interpretation follows the implementation of GSVM where items of no interest have no influence on the bundle
     *     values.
     *     
     * The interpretation which was used to implement the legacy version of LVSM is stated as Note II. In addition to this note 
     * you may notice that licenses where a bidder has no interest in, do not only affect the computation of maximally connected 
     * subsets of licenses, but also increases the synergies in the legacy version. I.e. assume a bundle of 3 licenses DEJ. The 
     * bidder is only interested in licenses E and J. (see table 1 in Scheffel et al. (2010)). In the current version no synergies 
     * will be applied to this package. The legacy version will apply synergies for 3 licenses. Note II could also be interpreted 
     * towards that only synergies for 2 licenses are applied and the license of no interest (D) is only used to compute maximally 
     * connected subpackages. This last interpretation is not implemented in any version of LSVM in sats.
     * 
     * Note II: To our knowledge, Scheffel et al. (2010) do not specify the behavior in case a license is added which the
     * bidder does not have any interest in. This one license could connect two subsets of items which the
     * bidder is interested in, creating one larger subset, thus increasing the bonus for adjacent items.
     * A possible extension of the model is to adjust the bonus accordingly in this situation.
     * Currently, SATS doesn't prevent this behavior, sticking to the original model.
     */
    public double calculateFactor(int size) {
        return 1 + (LSVM_A / (100 * (1 + Math.exp(LSVM_B - size))));
    }

    @Override
    public LinkedHashSet<Bundle> getBestBundles(Prices prices, int maxNumberOfBundles, boolean allowNegative) {
        LSVMStandardMIP mip = new LSVMStandardMIP(world, Lists.newArrayList(this));
        mip.setMipInstrumentation(getMipInstrumentation());
        mip.setPurpose(MipInstrumentation.MipPurpose.DEMAND_QUERY.name());
        Variable priceVar = new Variable("p", VarType.DOUBLE, 0, MIP.MAX_VALUE);
        mip.getMIP().add(priceVar);
        mip.getMIP().addObjectiveTerm(-1, priceVar);
        Constraint price = new Constraint(CompareType.EQ, 0);
        price.addTerm(-1, priceVar);
        for (LSVMLicense license : world.getLicenses()) {
        	Map<Integer, Variable> xVariables = mip.getXVariables(this, license);
        	for (Variable xVariable : xVariables.values()) {
        		price.addTerm(prices.getPrice(Bundle.of(license)).getAmount().doubleValue(), xVariable);
        	}

        }
        mip.getMIP().add(price);
        
        mip.setEpsilon(DEFAULT_DEMAND_QUERY_EPSILON);
        mip.setTimeLimit(DEFAULT_DEMAND_QUERY_TIME_LIMIT);
        
        if(this.getWorld().isLegacyLSVM()) {
        	maxNumberOfBundles = Math.min(maxNumberOfBundles, (int) Math.pow(2, this.getWorld().getNumberOfGoods()));
        } else {
        	maxNumberOfBundles = Math.min(maxNumberOfBundles, (int) Math.pow(2, this.getProximity().size()));
        }
        
        List<Allocation> optimalAllocations = mip.getBestAllocations(maxNumberOfBundles, allowNegative);

        LinkedHashSet<Bundle> result = optimalAllocations.stream()
                .map(allocation -> allocation.allocationOf(this).getBundle())
                .filter(bundle -> allowNegative || getUtility(bundle, prices).signum() > -1)
                .collect(Collectors.toCollection(LinkedHashSet::new));
        if (result.isEmpty()) result.add(Bundle.EMPTY);
        return result;
    }

    @Override
    public String getDescription() {
        return description;
    }

}
