// Generate pi0 -> e+ e- gamma Dalitz decays at rest using Geant4's
// G4DalitzDecayChannel (the same model that decays GENIE pi0s in the larg4
// detector simulation), and dump the daughter four-momenta (MeV) to a CSV.
//
// Because we call the decay channel directly with the pi0 at rest, the products
// are already in the pi0 rest frame -- no geometry, physics list, or tracking is
// needed, so this generates millions of decays cheaply.
//
// Usage:
//   g4_dalitz_gen <n_events> <out.csv> [seed]

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include "G4DalitzDecayChannel.hh"
#include "G4DecayProducts.hh"
#include "G4DynamicParticle.hh"
#include "G4Electron.hh"
#include "G4Gamma.hh"
#include "G4PionZero.hh"
#include "G4Positron.hh"
#include "G4SystemOfUnits.hh"
#include "G4ThreeVector.hh"
#include "Randomize.hh"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <n_events> <out.csv> [seed]\n";
        return 1;
    }
    const long nEvents = std::atol(argv[1]);
    const std::string outFile = argv[2];
    const long seed = (argc > 3) ? std::atol(argv[3]) : 12345;

    CLHEP::HepRandom::setTheSeed(seed);

    // register the particle definitions in the G4 particle table
    G4ParticleDefinition* pi0 = G4PionZero::Definition();
    G4Gamma::Definition();
    G4Electron::Definition();
    G4Positron::Definition();
    const G4double pi0Mass = pi0->GetPDGMass();

    // BR = 1.0: every call produces a Dalitz decay (pi0 -> gamma e- e+)
    G4DalitzDecayChannel dalitz("pi0", 1.0, "e-", "e+");

    std::ofstream out(outFile);
    out << "E_g,px_g,py_g,pz_g,"
           "E_ep,px_ep,py_ep,pz_ep,"
           "E_em,px_em,py_em,pz_em\n";
    out.precision(9);

    long written = 0;
    for (long i = 0; i < nEvents; ++i) {
        G4DecayProducts* prod = dalitz.DecayIt(pi0Mass);  // products in pi0 rest frame
        if (!prod) continue;

        double pg[4] = {0}, pep[4] = {0}, pem[4] = {0};
        bool okg = false, okep = false, okem = false;
        const G4int np = prod->entries();
        for (G4int j = 0; j < np; ++j) {
            G4DynamicParticle* d = (*prod)[j];
            const G4int pdg = d->GetDefinition()->GetPDGEncoding();
            const G4ThreeVector p = d->GetMomentum();   // MeV
            const G4double E = d->GetTotalEnergy();      // MeV
            double* t = nullptr;
            if (pdg == 22) { t = pg; okg = true; }
            else if (pdg == -11) { t = pep; okep = true; }
            else if (pdg == 11) { t = pem; okem = true; }
            if (t) { t[0] = E; t[1] = p.x(); t[2] = p.y(); t[3] = p.z(); }
        }
        if (okg && okep && okem) {
            out << pg[0] << "," << pg[1] << "," << pg[2] << "," << pg[3] << ","
                << pep[0] << "," << pep[1] << "," << pep[2] << "," << pep[3] << ","
                << pem[0] << "," << pem[1] << "," << pem[2] << "," << pem[3] << "\n";
            ++written;
        }
        delete prod;
    }
    out.close();

    std::cout << "generated " << written << " / " << nEvents
              << " pi0 Dalitz decays -> " << outFile << std::endl;
    return 0;
}
