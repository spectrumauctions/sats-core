/**
 * Copyright by Michael Weiss, weiss.michael@gmx.ch
 * <p>
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.spectrumauctions.sats.core.api;

import com.google.common.base.Preconditions;
import org.spectrumauctions.sats.core.bidfile.FileWriter;
import org.spectrumauctions.sats.core.bidlang.BiddingLanguage;
import org.spectrumauctions.sats.core.model.DefaultModel;
import org.spectrumauctions.sats.core.model.SATSBidder;
import org.spectrumauctions.sats.core.model.UnsupportedBiddingLanguageException;
import org.spectrumauctions.sats.core.util.file.FilePathUtils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.NoSuchElementException;

/**
 * @author Michael Weiss
 */
public abstract class ModelCreator {

    private final boolean oneFile;
    private final FileType fileType;
    private final boolean generic;
    private final int bidsPerBidder;
    private final long worldSeed;
    private final long populationSeed;
    private final BiddingLanguageEnum lang;

    private final boolean storeWorldSerialization;
    private SeedType seedType;
    private long superSeed;

    protected ModelCreator(Builder builder) {
        super();
        this.oneFile = builder.oneFile;
        this.fileType = builder.fileType;
        this.generic = builder.generic;
        this.bidsPerBidder = builder.bidsPerBidder;
        this.seedType = builder.seedType;
        this.superSeed = builder.superSeed;
        this.worldSeed = builder.worldSeed;
        this.populationSeed = builder.populationSeed;
        this.storeWorldSerialization = builder.storeWorldSerialization;
        this.lang = builder.lang;
    }

    public boolean isOneFile() {
        return oneFile;
    }

    public FileType getFileType() {
        return fileType;
    }

    public boolean isGeneric() {
        return generic;
    }

    public int getBidsPerBidder() {
        return bidsPerBidder;
    }

    public long getWorldSeed() {
        return worldSeed;
    }


    public long getPopulationSeed() {
        return populationSeed;
    }

    public boolean isStoreWorldSerialization() {
        return storeWorldSerialization;
    }

    public abstract PathResult generateResult(File outputFolder) throws UnsupportedBiddingLanguageException, IOException, IllegalConfigException;

    protected PathResult appendTopLevelParamsAndSolve(DefaultModel<?, ?> model, File outputFolder) throws UnsupportedBiddingLanguageException, IOException, IllegalConfigException {

        Collection<? extends SATSBidder> bidders;
        if (seedType == SeedType.INDIVIDUALSEED) {
            bidders = model.createNewWorldAndPopulation(worldSeed, populationSeed);
        } else if (seedType == SeedType.NOSEED) {
            bidders = model.createNewWorldAndPopulation();
        } else if (seedType == SeedType.SUPERSEED) {
            bidders = model.createNewWorldAndPopulation(superSeed);
        } else {
            throw new IllegalConfigException("Seed type unknown");
        }
        FileWriter writer = FileType.getFileWriter(fileType, outputFolder);

        FilePathUtils filePathUtils = FilePathUtils.getInstance();
        File instanceFolder = filePathUtils.worldFolderPath(bidders.stream().findAny().orElseThrow(NoSuchElementException::new).getWorldId());
        PathResult result;
        if (generic) {
            @SuppressWarnings("unchecked")
            Class<? extends BiddingLanguage> langClass = BiddingLanguageEnum.getXORQLanguage(lang);
            if (oneFile) {
                Collection<BiddingLanguage> languages = new ArrayList<>();
                for (SATSBidder bidder : bidders) {
                    if (seedType == SeedType.SUPERSEED) {
                        languages.add(bidder.getValueFunction(langClass, superSeed));
                    } else {
                        languages.add(bidder.getValueFunction(langClass));
                    }
                }
                File valueFile = writer.writeMultiBidderXORQ(languages, bidsPerBidder, "satsvalue");
                result = new PathResult(storeWorldSerialization, instanceFolder);
                result.addValueFile(valueFile);
                return result;
            } else {
                Collection<BiddingLanguage> languages = new ArrayList<>();
                String zipId = String.valueOf(new Date().getTime());
                File folder = new File(writer.getFolder().getAbsolutePath().concat(File.separator).concat(zipId));
                folder.mkdir();
                for (SATSBidder bidder : bidders) {
                    BiddingLanguage valueFunction;
                    if (seedType == SeedType.SUPERSEED) {
                        valueFunction = bidder.getValueFunction(langClass, superSeed);
                    } else {
                        valueFunction = bidder.getValueFunction(langClass);
                    }
                    writer.writeSingleBidderXORQ(valueFunction, bidsPerBidder, zipId.concat(File.separator).concat("satsvalue"));
                }
                result = new PathResult(storeWorldSerialization, instanceFolder);
                result.addValueFile(folder);
                return result;
            }
        } else {
            @SuppressWarnings("unchecked")
            Class<? extends BiddingLanguage> langClass = BiddingLanguageEnum.getXORLanguage(lang);
            if (oneFile) {
                Collection<BiddingLanguage> languages = new ArrayList<>();
                for (SATSBidder bidder : bidders) {
                    if (seedType == SeedType.SUPERSEED) {
                        languages.add(bidder.getValueFunction(langClass, superSeed));
                    } else {
                        languages.add(bidder.getValueFunction(langClass));
                    }
                }
                File valueFile = writer.writeMultiBidderXOR(languages, bidsPerBidder, "satsvalue");
                result = new PathResult(storeWorldSerialization, instanceFolder);
                result.addValueFile(valueFile);
                return result;
            } else {
                String zipId = String.valueOf(new Date().getTime());
                File folder = new File(writer.getFolder().getAbsolutePath().concat(File.separator).concat(zipId));
                folder.mkdir();
                for (SATSBidder bidder : bidders) {
                    BiddingLanguage language;
                    if (seedType == SeedType.SUPERSEED) {
                        language = bidder.getValueFunction(langClass, superSeed);
                    } else {
                        language = bidder.getValueFunction(langClass);
                    }
                    writer.writeSingleBidderXOR(language, bidsPerBidder, zipId.concat(File.separator).concat("satsvalue"));
                }
                result = new PathResult(storeWorldSerialization, instanceFolder);
                result.addValueFile(folder);
                return result;
            }
        }
    }


    public static abstract class Builder {

        private BiddingLanguageEnum lang;
        private boolean storeWorldSerialization;
        private long populationSeed;
        private long superSeed;
        private long worldSeed;
        private SeedType seedType;
        private int bidsPerBidder;
        private boolean generic;
        private FileType fileType;
        private boolean oneFile;

        public Builder() {
            this.lang = BiddingLanguageEnum.RANDOM;
            storeWorldSerialization = false;
            seedType = SeedType.NOSEED;
            bidsPerBidder = 100;
            generic = false;
            fileType = FileType.CATS;
            oneFile = true;
        }

        public abstract ModelCreator build();

        public boolean isStoreWorldSerialization() {
            return storeWorldSerialization;
        }

        public void setStoreWorldSerialization(boolean storeWorldSerialization) {
            if (storeWorldSerialization) {
                throw new UnsupportedOperationException("Storing through API not yet supported");
            }
            this.storeWorldSerialization = storeWorldSerialization;
        }

        public long getPopulationSeed() {
            return populationSeed;
        }

        public void setPopulationSeed(long populationSeed) {
            this.populationSeed = populationSeed;
        }

        public long getWorldSeed() {
            return worldSeed;
        }

        public void setWorldSeed(long worldSeed) {
            this.worldSeed = worldSeed;
        }

        public int getBidsPerBidder() {
            return bidsPerBidder;
        }

        public void setSuperSeed(long superSeed) {
            this.superSeed = superSeed;
        }

        public void setSeedType(SeedType seedType) {
            this.seedType = seedType;
        }

        public void setLang(BiddingLanguageEnum lang) {
            this.lang = lang;
        }

        public void setBidsPerBidder(int bidsPerBidder) throws IllegalConfigException {
            try {
                Preconditions.checkArgument(bidsPerBidder > 0, "%s is not a valid number of bids per bidder", bidsPerBidder);
            } catch (IllegalArgumentException e) {
                throw new IllegalConfigException(e.getMessage());
            }
            this.bidsPerBidder = bidsPerBidder;
        }

        public boolean isGeneric() {
            return generic;
        }

        public void setGeneric(boolean generic) {
            this.generic = generic;
        }

        public FileType getFileType() {
            return fileType;
        }

        public void setFileType(FileType fileType) {
            this.fileType = fileType;
        }

        public boolean isOneFile() {
            return oneFile;
        }

        public void setOneFile(boolean oneFile) {
            this.oneFile = oneFile;
        }

    }
}