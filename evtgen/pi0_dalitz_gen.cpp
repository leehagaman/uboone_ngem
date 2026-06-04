// Generate pi0 -> e+ e- gamma Dalitz decays at rest with EvtGen and write the
// daughter four-momenta (in MeV) to a CSV file, for comparison against the
// Geant4 BNB-overlay truth Dalitz sample.
//
// Usage:
//   pi0_dalitz_gen <DECAY.DEC> <evt.pdl> <user.dec> <n_events> <out.csv>
//
// The pi0 is created at rest, so the output four-momenta are already in the
// pi0 rest frame.

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <list>
#include <string>

#include "EvtGen/EvtGen.hh"
#include "EvtGenBase/EvtAbsRadCorr.hh"
#include "EvtGenBase/EvtDecayBase.hh"
#include "EvtGenBase/EvtPDL.hh"
#include "EvtGenBase/EvtParticle.hh"
#include "EvtGenBase/EvtParticleFactory.hh"
#include "EvtGenBase/EvtRandom.hh"
#include "EvtGenBase/EvtRandomEngine.hh"
#include "EvtGenBase/EvtVector4R.hh"

#include "EvtGenExternal/EvtExternalGenList.hh"

#ifdef EVTGEN_CPP11
#include "EvtGenBase/EvtMTRandomEngine.hh"
#endif
#include "EvtGenBase/EvtSimpleRandomEngine.hh"

int main(int argc, char* argv[]) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0]
                  << " <DECAY.DEC> <evt.pdl> <user.dec> <n_events> <out.csv>\n";
        return 1;
    }
    const std::string decayFile = argv[1];
    const std::string pdlFile = argv[2];
    const std::string userDecay = argv[3];
    const long nEvents = std::atol(argv[4]);
    const std::string outFile = argv[5];

    // ---- random engine ----
#ifdef EVTGEN_CPP11
    EvtRandomEngine* eng = new EvtMTRandomEngine();
#else
    EvtRandomEngine* eng = new EvtSimpleRandomEngine();
#endif
    EvtRandom::setRandomEngine(eng);

    // ---- external models (Photos, etc.) ----
    EvtAbsRadCorr* radCorrEngine = nullptr;
    std::list<EvtDecayBase*> extraModels;
    EvtExternalGenList genList;
    radCorrEngine = genList.getPhotosModel();
    extraModels = genList.getListOfModels();

    // ---- generator ----
    EvtGen myGenerator(decayFile.c_str(), pdlFile.c_str(), eng, radCorrEngine,
                       &extraModels);
    // override pi0 decay to force the Dalitz channel
    myGenerator.readUDecay(userDecay.c_str());

    const EvtId PI0 = EvtPDL::getId("pi0");
    const EvtId EP = EvtPDL::getId("e+");
    const EvtId EM = EvtPDL::getId("e-");
    const EvtId GAMMA = EvtPDL::getId("gamma");
    const double pi0Mass = EvtPDL::getMass(PI0);

    std::ofstream out(outFile);
    out << "E_g,px_g,py_g,pz_g,"
           "E_ep,px_ep,py_ep,pz_ep,"
           "E_em,px_em,py_em,pz_em\n";
    out.precision(9);

    auto writeP4 = [&out](const EvtVector4R& p, bool last) {
        // EvtGen units are GeV; convert to MeV
        out << p.get(0) * 1000.0 << "," << p.get(1) * 1000.0 << ","
            << p.get(2) * 1000.0 << "," << p.get(3) * 1000.0
            << (last ? "\n" : ",");
    };

    long written = 0;
    for (long i = 0; i < nEvents; ++i) {
        // pi0 at rest: p = (m, 0, 0, 0)
        EvtVector4R pInit(pi0Mass, 0.0, 0.0, 0.0);
        EvtParticle* root = EvtParticleFactory::particleFactory(PI0, pInit);
        myGenerator.generateDecay(root);

        if (root->getNDaug() != 3) {
            root->deleteTree();
            continue;
        }

        EvtVector4R pg, pep, pem;
        bool okg = false, okep = false, okem = false;
        for (size_t d = 0; d < root->getNDaug(); ++d) {
            EvtParticle* dau = root->getDaug(d);
            const EvtVector4R p4 = dau->getP4Lab();  // pi0 at rest -> rest frame
            if (dau->getId() == GAMMA) { pg = p4; okg = true; }
            else if (dau->getId() == EP) { pep = p4; okep = true; }
            else if (dau->getId() == EM) { pem = p4; okem = true; }
        }
        if (okg && okep && okem) {
            writeP4(pg, false);
            writeP4(pep, false);
            writeP4(pem, true);
            ++written;
        }
        root->deleteTree();
    }

    out.close();
    std::cout << "generated " << written << " / " << nEvents
              << " pi0 Dalitz decays -> " << outFile << std::endl;

    delete eng;
    return 0;
}
