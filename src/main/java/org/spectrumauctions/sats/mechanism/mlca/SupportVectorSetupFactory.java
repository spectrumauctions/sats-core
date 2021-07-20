package org.spectrumauctions.sats.mechanism.mlca;

import java.math.BigDecimal;
import java.util.Map;

import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.SupportVectorSetup;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.Kernel;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelDotProductExponential;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelDotProductPolynomial;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelGaussian;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelLinear;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelQuadratic;
import org.marketdesignresearch.mechlib.mechanism.auctions.mlca.svr.kernels.KernelType;

public enum SupportVectorSetupFactory {
	INSTANCE;
	
	// Linear and Quadratic kernels tuned for MLCA with Q_max = 100, Q_init = 50, Q_round = 4
	private Map<KernelType,Kernel> gsvmDefaultKernels = Map.of(
			KernelType.Linear, new KernelLinear(0d, 1d), 
			KernelType.Quadratic, new KernelQuadratic(0d, 1d, 0.0078125d),
			KernelType.Polynomial, new KernelDotProductPolynomial(new double[] {0d, 1d, 0.0078125d}),
			KernelType.Exponential, new KernelDotProductExponential(10d, 10d),
			KernelType.Gaussian, new KernelGaussian(20d, 10d));
	
	// Linear and Quadratic SVR Setup tuned for MLCA with Q_max = 100, Q_init = 50, Q_round = 4
	private Map<KernelType,SupportVectorSetup> gsvmTunedDefaultSetups = Map.of(
			KernelType.Linear, new SupportVectorSetup(100000000d, 0.00001d, BigDecimal.valueOf(100),true,1e-8, this.gsvmDefaultKernels.get(KernelType.Linear)),
			KernelType.Quadratic, new SupportVectorSetup(1000000d, 0.00001d, BigDecimal.valueOf(100),true,1e-8, this.gsvmDefaultKernels.get(KernelType.Quadratic))
					);
	
	//Linear and Quadratic kernels tuned for MLCA with Q_max = 500, Q_init = 50, Q_round = 4
	private Map<KernelType,Kernel> lsvmDefaultKernels = Map.of(
			KernelType.Linear, new KernelLinear(0d,1d),
			KernelType.Quadratic, new KernelQuadratic(0d, 1d, 0.03125d),
			KernelType.Polynomial, new KernelDotProductPolynomial(new double[] {0d, 1d, 0.03125d}),
			KernelType.Exponential, new KernelDotProductExponential(5d, 10d),
			KernelType.Gaussian, new KernelGaussian(10d,10d));
	
	private Map<KernelType,SupportVectorSetup> lsvmTunedDefaultSetups = Map.of(
			KernelType.Linear, new SupportVectorSetup(1000000d,0.00001d,BigDecimal.valueOf(100),true,1e-8,this.lsvmDefaultKernels.get(KernelType.Linear)),
			KernelType.Quadratic, new SupportVectorSetup(100000d,0.00001d,BigDecimal.valueOf(100),true,1e-8,this.lsvmDefaultKernels.get(KernelType.Quadratic))
			);
			
	
	private Map<KernelType,Kernel> mrvmDefaultKernels = Map.of(
			KernelType.Linear, new KernelLinear(0d,1d),
			KernelType.Quadratic, new KernelQuadratic(0d,1d,0.001d),
			KernelType.Polynomial, new KernelDotProductPolynomial(new double[] {0d,1d,0.001d}),
			KernelType.Exponential, new KernelDotProductExponential(100d, 100d),
			KernelType.Gaussian, new KernelGaussian(100d, 100d));
	
	public SupportVectorSetup createDefaultGSVMSupportVectorSetup(KernelType kernelType) {
		return gsvmTunedDefaultSetups.getOrDefault(kernelType, 
				new SupportVectorSetup(1000000d, 0.00001d, BigDecimal.valueOf(100),true,1e-8, this.gsvmDefaultKernels.get(kernelType)));
	}
	
	public SupportVectorSetup createDefaultLSVMSupportVectorSetup(KernelType kernelType) {
		return lsvmTunedDefaultSetups.getOrDefault(kernelType,
				new SupportVectorSetup(100000d,0.00001d,BigDecimal.valueOf(100),true,1e-8,this.lsvmDefaultKernels.get(kernelType)));
	}
	
	public SupportVectorSetup createDefaultMRVMSupportVectorSetup(KernelType kernelType) {
		return new SupportVectorSetup(1000d,0.00001d,BigDecimal.ONE.divide(BigDecimal.valueOf(1000000)),true,1e-8,this.mrvmDefaultKernels.get(kernelType));
	}
}
