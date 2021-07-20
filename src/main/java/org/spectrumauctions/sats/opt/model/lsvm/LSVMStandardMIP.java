package org.spectrumauctions.sats.opt.model.lsvm;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import edu.harvard.econcs.jopt.solver.ISolution;
import edu.harvard.econcs.jopt.solver.SolveParam;
import edu.harvard.econcs.jopt.solver.mip.CompareType;
import edu.harvard.econcs.jopt.solver.mip.Constraint;
import edu.harvard.econcs.jopt.solver.mip.VarType;
import edu.harvard.econcs.jopt.solver.mip.Variable;
import lombok.EqualsAndHashCode;
import org.marketdesignresearch.mechlib.core.Allocation;
import org.marketdesignresearch.mechlib.core.BidderAllocation;
import org.marketdesignresearch.mechlib.core.Bundle;
import org.marketdesignresearch.mechlib.core.Good;
import org.marketdesignresearch.mechlib.core.allocationlimits.AllocationLimitConstraint;
import org.marketdesignresearch.mechlib.core.bid.bundle.BundleExactValueBids;
import org.marketdesignresearch.mechlib.core.bid.bundle.BundleValueBids;
import org.marketdesignresearch.mechlib.core.bidder.Bidder;
import org.marketdesignresearch.mechlib.instrumentation.MipInstrumentation;
import org.marketdesignresearch.mechlib.metainfo.MetaInfo;
import org.spectrumauctions.sats.core.model.cats.graphalgorithms.Vertex;
import org.spectrumauctions.sats.core.model.gsvm.GSVMBidder;
import org.spectrumauctions.sats.core.model.lsvm.LSVMBidder;
import org.spectrumauctions.sats.core.model.lsvm.LSVMGrid;
import org.spectrumauctions.sats.core.model.lsvm.LSVMLicense;
import org.spectrumauctions.sats.core.model.lsvm.LSVMWorld;
import org.spectrumauctions.sats.opt.model.ModelMIP;

import java.math.BigDecimal;
import java.util.*;
import java.util.stream.Collectors;

/**
 * This class implements a mip finding the efficient allocation in an instance
 * of the Local-Synergy Value Model
 *
 * @author Nicolas Küchler
 */
public class LSVMStandardMIP extends ModelMIP {

	private Map<LSVMBidder, Map<LSVMLicense, Double>> valueMap;

	private List<LSVMBidder> population;

	private LSVMWorld world;

	private Map<LSVMBidder, Map<LSVMLicense, Map<Integer, Variable>>> aVariables;
	private Map<LSVMBidder, Map<Edge, Map<Integer, Variable>>> eVariables;

	private Collection<Collection<Variable>> variableSetsOfInterest = new HashSet<>();

	private Map<Edge, Set<Integer>> validPathLengths = new HashMap<>();

	public LSVMStandardMIP(List<LSVMBidder> population) {
		this(population.iterator().next().getWorld(), population);
	}

	public LSVMStandardMIP(LSVMWorld world, List<LSVMBidder> population) {
		this.world = world;
		this.population = population;

		// init MIP
		getMIP().setObjectiveMax(true);
		getMIP().setSolveParam(SolveParam.TIME_LIMIT, 3600.0);

		initBaseValues();
		initA();
		initEdge();
		initE();

		buildObjectiveTerm();
		buildSupplyEvalConstraints();
		buildEdgeSupplyConstraints();
		buildNeighbourConstraints();
		buildEdgeConstraints();
		buildTauConstraints();
		buildAllocationLimits();
		if(!this.world.isLegacyLSVM()) {
			buildLicenceRestrictions();
		}
	}

	@Override
	public ModelMIP getMIPWithout(Bidder bidder) {
		LSVMBidder lsvmBidder = (LSVMBidder) bidder;
		Preconditions.checkArgument(population.contains(lsvmBidder));
		return new LSVMStandardMIP(population.stream().filter(b -> !b.equals(lsvmBidder)).collect(Collectors.toList()));
	}

	@Override
	protected Allocation adaptMIPResult(ISolution solution) {

		Map<Bidder, BidderAllocation> allocationMap = new HashMap<>();

		for (LSVMBidder bidder : population) {
			Set<LSVMLicense> licenseSet = new HashSet<>();
            for (LSVMLicense license : world.getLicenses()) {
                for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                    if (solution.getValue(aVariables.get(bidder).get(license).get(tau)) > 0) {
                        licenseSet.add(license);
                    }
                }
            }
			Bundle bundle = Bundle.of(licenseSet);
            if (!Bundle.EMPTY.equals(bundle)) {
				allocationMap.put(bidder, new BidderAllocation(bidder.calculateValue(bundle), bundle, new HashSet<>()));
			}
		}

		MetaInfo metaInfo = new MetaInfo();
		metaInfo.setNumberOfMIPs(1);
		metaInfo.setMipSolveTime(solution.getSolveTime());

		return new Allocation(allocationMap, new BundleExactValueBids(), metaInfo);
	}

    public Map<Integer, Variable> getXVariables(LSVMBidder bidder, LSVMLicense license) {
        for (LSVMBidder b : population) {
            if (b.equals(bidder)) {
                for (LSVMLicense l : world.getLicenses()) {
                    if (l.equals(license)) {
                        return aVariables.get(b).get(l);
                    }
                }
            }
        }
        return new HashMap<>();
    }

	@Override
	public ModelMIP copyOf() {
		return new LSVMStandardMIP(population);
	}

	@Override
	protected Collection<Collection<Variable>> getVariablesOfInterest() {
		return variableSetsOfInterest;
	}

	@Override
	public PoolMode getSolutionPoolMode() {
		return PoolMode.MODE_3;
	}


	private void buildObjectiveTerm() {
		for (LSVMBidder bidder : population) {
            for (LSVMLicense license : world.getLicenses()) {
                for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                    double value = calculateComplementarityMarkup(tau + 1, bidder) * valueMap.get(bidder).get(license);
                    getMIP().addObjectiveTerm(value, aVariables.get(bidder).get(license).get(tau));
                }
            }
		}
	}

	private void buildSupplyEvalConstraints() {
		for (LSVMLicense license : world.getLicenses()) {
			Constraint constraint = new Constraint(CompareType.LEQ, 1);
			for (LSVMBidder bidder : population) {
                for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                    constraint.addTerm(1, aVariables.get(bidder).get(license).get(tau));
                }
			}
			getMIP().add(constraint);
		}
	}

	private void buildEdgeSupplyConstraints() {
		for (Map.Entry<Edge, Set<Integer>> entry : validPathLengths.entrySet()) {
			Constraint constraint = new Constraint(CompareType.LEQ, 1);
			for (LSVMBidder bidder : population) {
                for (int c = 0; c < world.getLicenses().size(); c++) {
                    if (entry.getValue().contains(c + 1)) {
                        constraint.addTerm(1, eVariables.get(bidder).get(entry.getKey()).get(c));
                    }
                }
			}
			getMIP().add(constraint);
		}
	}

	private void buildNeighbourConstraints() {
		for (LSVMBidder bidder : population) {
            for (Map.Entry<Edge, Set<Integer>> entry : validPathLengths.entrySet()) {
                Edge edge = entry.getKey();
                for (int c = 1; c < world.getLicenses().size(); c++) { // only for c > 1
                    if (entry.getValue().contains(c + 1)) {
                        Constraint constraint = new Constraint(CompareType.GEQ, 0);
                        constraint.addTerm(-1, eVariables.get(bidder).get(edge).get(c));
                        for (LSVMLicense x : n(gMin(edge))) {
                            LSVMLicense y = gMax(edge);
                            if (!x.equals(y)) {
                                Edge e = fInv(x, y);
                                if (validPathLengths.get(e).contains(c)) {
                                    constraint.addTerm(1, eVariables.get(bidder).get(e).get(c - 1));
                                }
                            }
                        }
                        getMIP().add(constraint);
                    }
                }
            }
		}
	}

	private void buildEdgeConstraints() {
		for (LSVMBidder bidder : population) {
            for (Map.Entry<Edge, Set<Integer>> entry : validPathLengths.entrySet()) {
                Edge edge = entry.getKey();
                Constraint constraint = new Constraint(CompareType.GEQ, 0);
                for (int c = 0; c < world.getLicenses().size(); c++) {
                    if (entry.getValue().contains(c + 1)) {
                        constraint.addTerm(-2, eVariables.get(bidder).get(edge).get(c));
                    }
                }
                for (LSVMLicense license : f(edge)) {
                    for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                        constraint.addTerm(1, aVariables.get(bidder).get(license).get(tau));
                    }
                }
                getMIP().add(constraint);
            }
		}
	}

	private void buildTauConstraints() {
		for (LSVMBidder bidder : population) {
            for (LSVMLicense license : world.getLicenses()) {
                Constraint constraint = new Constraint(CompareType.LEQ, 1);
                for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                    constraint.addTerm(tau + 1, aVariables.get(bidder).get(license).get(tau));
                }

                for (LSVMLicense other : world.getLicenses()) {
                    if (!other.equals(license)) {
                        for (int c = 0; c < world.getLicenses().size(); c++) {
                            Edge edge = fInv(other, license);
                            if (validPathLengths.get(edge).contains(c + 1)) {
                                constraint.addTerm(-1, eVariables.get(bidder).get(edge).get(c));
                            }
                        }
                    }
                }
                getMIP().add(constraint);
            }
        }
	}
	
	private void buildLicenceRestrictions() {
		for (LSVMBidder bidder : population) {
			for (LSVMLicense license : world.getLicenses()) {
	        	Map<Integer, Variable> xVariables = this.getXVariables(bidder, license);
	        	for (Variable xVariable : xVariables.values()) {
	        		if(!bidder.getProximity().contains(license)) {
	        			xVariable.setUpperBound(0);
	        		}
	        	}

	        }
		}
	}

	private void initBaseValues() {
	    valueMap = new HashMap<>();
		for (LSVMBidder bidder : population) {
		    valueMap.put(bidder, new HashMap<>());
            for (LSVMLicense license : world.getLicenses()) {
                valueMap.get(bidder).put(license, bidder.getBaseValues().getOrDefault(license.getLongId(), BigDecimal.ZERO).doubleValue());
            }
		}
	}

	private void initA() {
	    aVariables = new HashMap<>();
		for (LSVMBidder bidder : population) {
		    aVariables.put(bidder, new HashMap<>());
            for (LSVMLicense license : world.getLicenses()) {
            	Collection<Variable> xVariables = new HashSet<>();
                aVariables.get(bidder).put(license, new HashMap<>());
                for (int tau = 0; tau < world.getLicenses().size(); tau++) {
                    Variable var = new Variable(String.format("A_i[%d]j[%d]tau[%d]", (int) bidder.getLongId(), (int) license.getLongId(), tau), VarType.BOOLEAN, 0, 1);
                    getMIP().add(var);
                    aVariables.get(bidder).get(license).put(tau, var);
                    xVariables.add(var);
                }
                variableSetsOfInterest.add(xVariables);
            }
		}
	}

	private void initEdge() {
		for (LSVMLicense l1 : world.getLicenses()) {
			for (LSVMLicense l2 : world.getLicenses()) {
                Edge edge = new Edge(l1, l2);
			    if (!validPathLengths.containsKey(edge)) {
                    buildValidPathLength(edge);
                }
			}
		}
	}

	private void buildValidPathLength(Edge edge) {
		LSVMGridGraph grid = new LSVMGridGraph(world.getGrid());
		Set<Set<Vertex>> allPaths = grid.findAllPaths(grid.getVertex(edge.l1), grid.getVertex(edge.l2));

		List<Set<Vertex>> sizeOrderedPaths = allPaths.stream().sorted(Comparator.comparingInt(Set::size))
				.collect(Collectors.toList());

		List<Set<Vertex>> solution = new ArrayList<>();

		for (Set<Vertex> current : sizeOrderedPaths) {
			if (solution.stream().noneMatch(current::containsAll)) {
				solution.add(current);
			}
		}
		Set<Integer> valid = solution.stream().map(path -> path.size() - 1).collect(Collectors.toSet());
		validPathLengths.put(edge, valid);
	}
	
	private void buildAllocationLimits() {
		for(LSVMBidder bidder : this.population) {
			Map<Good, List<Variable>> bidderVariables = aVariables.get(bidder).entrySet().stream().collect(Collectors.toMap(e->e.getKey(), e-> new ArrayList<>(e.getValue().values()), (e1, e2) -> e1, LinkedHashMap::new));
			for(AllocationLimitConstraint alc : bidder.getAllocationLimit().getConstraints()) {
				this.getMIP().add(alc.createCPLEXConstraintWithMultiVarsPerGood(bidderVariables));
			}
		}
	}

	private void initE() {
	    eVariables = new HashMap<>();
		for (LSVMBidder bidder : population) {
		    eVariables.put(bidder, new HashMap<>());
            for (Map.Entry<Edge, Set<Integer>> entry : validPathLengths.entrySet()) {
                eVariables.get(bidder).put(entry.getKey(), new HashMap<>());
                for (int c = 0; c < world.getLicenses().size(); c++) {
                    if (entry.getValue().contains(c + 1)) {
                        Variable var = new Variable(String.format("E_i[%d]e[%s]c[%d]", (int) bidder.getLongId(), entry.getKey(), c), VarType.BOOLEAN, 0, 1);
                        getMIP().add(var);
                        eVariables.get(bidder).get(entry.getKey()).put(c, var);
                    }
                }
            }
		}
	}

	private double calculateComplementarityMarkup(int tau, LSVMBidder bidder) {
		if (tau < 1) {
			throw new IllegalArgumentException("Error: tau has to be >=1");
		}
		return bidder.calculateFactor(tau);
	}

	private LSVMLicense gMin(Edge e) {
		int neighbourCountL1 = n(e.l1).size();
		int neighbourCountL2 = n(e.l2).size();

		if (neighbourCountL1 <= neighbourCountL2) {
			return e.l1;
		} else {
			return e.l2;
		}
	}

	private LSVMLicense gMax(Edge e) {
		// return the other license on the edge not chosen by gMin
		Set<LSVMLicense> immutableSet = f(e);
		LSVMLicense minLicense = gMin(e);

		Set<LSVMLicense> mutableSet = immutableSet.stream().filter(l -> !l.equals(minLicense)).collect(Collectors.toSet());

		assert mutableSet.size() == 1;
		return mutableSet.iterator().next();
	}

	private Set<LSVMLicense> n(LSVMLicense license) {
		LSVMGrid grid = world.getGrid();
		return world.getLicenses().stream().filter(x -> grid.isNeighbor(x, license))
				.collect(Collectors.toSet());

	}

	private Set<LSVMLicense> f(Edge e) {
		return ImmutableSet.of(e.l1, e.l2);
	}

	private Edge fInv(LSVMLicense l1, LSVMLicense l2) {
		for (Edge edge : validPathLengths.keySet()) {
			if ((edge.l1 == l1 && edge.l2 == l2) || (edge.l1 == l2 && edge.l2 == l1)) {
				return edge;
			}
		}
		throw new IllegalStateException("Error: fInv edge not found");
	}

	@EqualsAndHashCode
	public class Edge {
		LSVMLicense l1;
		LSVMLicense l2;

		public Edge(LSVMLicense l1, LSVMLicense l2) {
		    if (l1.getLongId() > l2.getLongId()) {
                this.l1 = l1;
                this.l2 = l2;
            } else {
		        this.l2 = l1;
		        this.l1 = l2;
            }
		}

        @Override
        public String toString() {
            return "Edge(" + (int) l1.getLongId() + "," + (int) l2.getLongId() + ")";
        }
    }

}
