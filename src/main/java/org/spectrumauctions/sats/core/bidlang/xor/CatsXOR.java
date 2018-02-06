package org.spectrumauctions.sats.core.bidlang.xor;

import com.google.common.base.Preconditions;
import org.spectrumauctions.sats.core.model.Bundle;
import org.spectrumauctions.sats.core.model.cats.CATSBidder;
import org.spectrumauctions.sats.core.model.cats.CATSLicense;
import org.spectrumauctions.sats.core.model.cats.CATSWorld;
import org.spectrumauctions.sats.core.util.random.RNGSupplier;
import org.spectrumauctions.sats.core.util.random.UniformDistributionRNG;

import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.stream.Collectors;

/**
 * <p>The original CATS Regions model has a specific way to generate bids, which does not directly translate into our
 * iterator-based way of generating bids. This class provides an iterator that imitates the original bid-generation
 * technique. The first bundle of the provided iterator is the initial bundle which the following elements are based
 * on. The next bundles each have one license of the original bundle as a starting point and are extended so that they
 * have the same amount of licenses as the original bundle. In the CATS Regions model, they are called substitutable
 * bids/bundles.</p>
 * <p>In the original CATS Regions model, the number of substitutable bundles are capped at a certain number (see {@link
 * CATSWorld}), only considering the original bundle plus the up to X substitutable bundles with the highest value.
 * This can be imitated manually with the iterator, but the for convenience, this functionality is provided via the
 * method {@link #getCATSXORBids()}.</p>
 * <p>Per default, bundles that are created during the process which are not valid due to the original CATS Regions
 * constraints (not identical to any included license, within budget and minimal resale value) are not included in
 * {@link #getCATSXORBids()} and return null if generated via the iterator - so be sure to check for null values!
 * If you want to let the iterator find another bundle in case the current one is not valid (and abort after
 * #CATSIterator.MAX_RETRIES retries, because then there is most probably no other valid bundle), use the
 * {@link #iteratorWithoutNulls()}.</p>
 *
 * @author Fabio Isler
 */
public class CatsXOR implements XORLanguage<CATSLicense> {
    private Collection<CATSLicense> goods;
    private CATSBidder bidder;
    private RNGSupplier rngSupplier;
    private CATSWorld world;

    public CatsXOR(Collection<CATSLicense> goods, RNGSupplier rngSupplier, CATSBidder bidder) {
        this.goods = goods;
        this.bidder = bidder;
        this.rngSupplier = rngSupplier;
        this.world = goods.stream().findAny().orElseThrow(IllegalArgumentException::new).getWorld();
    }

    @Override
    public CATSBidder getBidder() {
        return bidder;
    }

    @Override
    public Iterator<XORValue<CATSLicense>> iterator() {
        return new CATSIterator(rngSupplier.getUniformDistributionRNG());
    }

    public Iterator<XORValue<CATSLicense>> iteratorWithoutNulls() {
        return new CATSIterator(rngSupplier.getUniformDistributionRNG(), false);
    }

    public Set<XORValue<CATSLicense>> getCATSXORBids() {
        TreeSet<XORValue<CATSLicense>> sortedSet = new TreeSet<>();
        Set<XORValue<CATSLicense>> result = new HashSet<>();

        Iterator<XORValue<CATSLicense>> iterator = new CATSIterator(rngSupplier.getUniformDistributionRNG());

        result.add(iterator.next()); // CATS always includes the original bundle

        // Fill the sorted set with all the elements that are not null
        while (iterator.hasNext()) {
            XORValue<CATSLicense> next = iterator.next();
            if (next != null) {
                sortedSet.add(next);
            }
        }

        // Get the most valuable elements from the substitutable bids
        for (int i = 0; i < world.getMaxSubstitutableBids() && !sortedSet.isEmpty(); i++) {
            XORValue<CATSLicense> val = sortedSet.first();
            if (!result.stream().map(XORValue::getLicenses).collect(Collectors.toList()).contains(val.getLicenses())) {
                result.add(val);
            }
            sortedSet.remove(val);
        }
        return result;
    }

    private class CATSIterator implements Iterator<XORValue<CATSLicense>> {
        private static final int MAX_RETRIES = 100;

        private final UniformDistributionRNG uniRng;
        private Queue<CATSLicense> originalLicenseQueue;
        private Bundle<CATSLicense> originalBundle;
        private double minValue;
        private double budget;
        private double minResaleValue;
        private int retries;
        private boolean acceptNulls;

        CATSIterator(UniformDistributionRNG uniRng, boolean acceptNulls) {
            Preconditions.checkArgument(world.getLicenses().size() == goods.size());
            this.uniRng = uniRng;
            this.minValue = 1e10;
            this.retries = 0;
            this.acceptNulls = acceptNulls;
        }

        CATSIterator(UniformDistributionRNG uniRng) {
            this(uniRng, true);
        }

        @Override
        public boolean hasNext() {
            if (originalBundle == null) return true;            // The first bundle has not been created yet
            int licensesLeftToChoose = goods.size() - originalBundle.size();
            return !(originalBundle.size() <= 1)                // The original bundle included only one license
                        && !originalLicenseQueue.isEmpty()      // We're not done yet with creating substitutable bundles
                        && licensesLeftToChoose > 0;
        }

        /**
         * @throws NoSuchElementException This exception is not only thrown if there was no {{@link #hasNext()} query
         *              before calling this function, but also if no valid bundle was found (not identical to the
         *              original bid and satisfying budget or min_resale_value constraints) after {{@link #MAX_RETRIES}
         *              retries.
         */
        @Override
        public XORValue<CATSLicense> next() throws NoSuchElementException {
            if (!hasNext())
                throw new NoSuchElementException();

            Bundle<CATSLicense> bundle = new Bundle<>();
            for (Map.Entry<Long, BigDecimal> entry : bidder.getPrivateValues().entrySet()) {
                if (entry.getValue().doubleValue() < minValue) minValue = entry.getValue().doubleValue();
            }

            if (originalLicenseQueue == null) {
                // We didn't construct an original bid yet
                WeightedRandomCollection<CATSLicense> weightedGoods = new WeightedRandomCollection<>(uniRng);
                goods.forEach(g -> {
                    double positivePrivateValue = (bidder.getPrivateValues().get(g.getId()).doubleValue() - minValue);
                    weightedGoods.add(positivePrivateValue, g);
                });
                CATSLicense first = weightedGoods.next();
                bundle.add(first);
                while (uniRng.nextDouble() <= world.getAdditionalLocation()) {
                    bundle.add(selectLicenseToAdd(bundle));
                }

                BigDecimal value = bidder.calculateValue(bundle);
                if (value.compareTo(BigDecimal.ZERO) < 0) return next(); // Restart bundle generation for this bidder

                budget = world.getBudgetFactor() * value.doubleValue();
                minResaleValue = world.getResaleFactor() * bundle.stream().mapToDouble(CATSLicense::getCommonValue).sum();
                originalLicenseQueue = new LinkedBlockingQueue<>(bundle);
                originalBundle = bundle;
                return new XORValue<>(bundle, value);
            } else {
                CATSLicense first = originalLicenseQueue.poll();
                bundle.add(first);
                while (bundle.size() < originalBundle.size()) {
                    CATSLicense toAdd = selectLicenseToAdd(bundle);
                    if (toAdd != null) bundle.add(toAdd);
                }
                BigDecimal value = bidder.calculateValue(bundle);
                double resaleValue = bundle.stream().mapToDouble(CATSLicense::getCommonValue).sum();
                if (value.doubleValue() >= 0 && value.doubleValue() <= budget
                        && resaleValue >= minResaleValue
                        && !bundle.equals(originalBundle)) {
                    retries = 0; // Found one - reset retries counter
                    return new XORValue<>(bundle, value);
                } else {
                    return handleNulls(first);
                }
            }
        }

        private XORValue<CATSLicense> handleNulls(CATSLicense first) throws NoSuchElementException {
            if (acceptNulls) {
                return null;
            }
            originalLicenseQueue.add(first); // Add this license to the original queue again
            if (hasNext() && ++retries < MAX_RETRIES) return next();
            else throw new NoSuchElementException("After " + retries + " retries, no other bundle was found " +
                    "that was not identical to the original bundle bid and is valid in terms of budget and " +
                    "min_resale_value constraints. \n" +
                    "Most likely, there are either almost no licenses to choose from or the original bundle is very" +
                    "small and highly valued, so that it's difficult to create another bundle that satisfies the" +
                    "constraints. Try again (maybe with a higher number of goods) or use the the iterator that handles" +
                    "this situation with null-values.");
        }

        private CATSLicense selectLicenseToAdd(Bundle<CATSLicense> bundle) {
            if (uniRng.nextDouble() <= world.getJumpProbability()) {
                if (goods.size() == bundle.size()) return null; // Prevent infinite loop if there is no other license
                CATSLicense randomLicense;
                do {
                    Iterator<CATSLicense> iterator = goods.iterator();
                    int index = uniRng.nextInt(goods.size());
                    for (int i = 0; i < index; i++) {
                        iterator.next();
                    }
                    randomLicense = iterator.next();
                } while(bundle.contains(randomLicense));

                return randomLicense;
            } else {
                WeightedRandomCollection<CATSLicense> neighbors = new WeightedRandomCollection<>(uniRng);
                // Filter the licenses that are not contained yet in the bundle and where there exists an edge to one
                // of the licenses in the bundle.
                goods.stream().filter(l -> !bundle.contains(l) && edgeExists(l, bundle))
                        .forEach(g -> {
                            double positivePrivateValue = bidder.getPrivateValues().get(g.getId()).doubleValue() - minValue;
                            neighbors.add(positivePrivateValue, g);
                        });
                if (neighbors.hasNext()) return neighbors.next();
                else return null;
            }
        }

        private boolean edgeExists(CATSLicense license, Bundle<CATSLicense> bundle) {
            for (CATSLicense l : bundle) {
                if (world.getGrid().isAdjacent(license.getVertex(), l.getVertex()))
                    return true;
            }
            return false;
        }

        /**
         * @see Iterator#remove()
         */
        @Override
        public void remove() {
            throw new UnsupportedOperationException();
        }
    }

    private class WeightedRandomCollection<T> implements Iterator<T> {
        private final NavigableMap<Double, T> map = new TreeMap<>();
        private final UniformDistributionRNG random;
        private double total = 0;

        public WeightedRandomCollection(UniformDistributionRNG random) {
            this.random = random;
        }

        public void add(double weight, T result) {
            total += weight;
            map.put(total, result);
        }

        @Override
        public boolean hasNext() {
            return !map.isEmpty();
        }

        public T next() {
            double value = random.nextDouble() * total;
            Map.Entry<Double, T> entry = map.ceilingEntry(value);
            if (entry == null)
                return null;
            return entry.getValue();
        }
    }

}
