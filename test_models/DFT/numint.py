import numpy
from pyscf import lib
from pyscf.dft import numint, rks, uks
from pyscf.dft.gen_grid import NBINS
from pyscf.dft.numint import (_dot_ao_ao, _dot_ao_ao_sparse, _format_uks_dm,
                              _scale_ao, _scale_ao_sparse, _tau_dot_sparse)

class NumIntWithLaplacian(numint.NumInt):

    def nr_rks(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):
        """
        Custom RKS integration that handles a functional returning (vrho, vsigma, vlapl, vtau).
        """
        xctype = self._xc_type(xc_code)

        make_rho, nset, nao = self._gen_rho_evaluator(mol, dms, hermi, with_lapl=True, grids=grids)

        ao_deriv = 2 
        
        if isinstance(dms, numpy.ndarray):
            dms_dtype = dms.dtype
        else:
            dms_dtype = numpy.result_type(*dms)

        nelec = numpy.zeros(nset)
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((nset, nao, nao), dtype=dms_dtype)

        for ao, mask, weight, coords in self.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho = make_rho(i, ao, mask, xctype)

                exc, vxc_tuple = self.eval_xc(xc_code, rho, spin=0, relativity=0, deriv=1)[:2]

                vrho, vsigma, vlapl, vtau = vxc_tuple

                den = rho[0] * weight
                nelec[i] += den.sum()
                excsum[i] += numpy.dot(den, exc)

                rho_std = numpy.concatenate((rho[:4], rho[5:6]), axis=0)
                
                vxc_std = (vrho, vsigma, None, vtau)
                
                vmat[i] += numint.eval_mat(mol, ao, weight, rho_std, vxc_std,
                                           non0tab=mask, xctype=xctype, spin=0)

                
                if vlapl is not None:
                    w_vlapl = weight * vlapl
                    
                    ao_lapl = ao[4] + ao[7] + ao[9]
                    
                    vp = numint._scale_ao(ao_lapl, w_vlapl)
                    
                    mat_term1 = numint._dot_ao_ao(mol, vp, ao[0], mask, shls_slice=None, ao_loc=None)

                    vmat[i] += mat_term1 + mat_term1.T
                    
                    for d in [1, 2, 3]:

                        vg = numint._scale_ao(ao[d], 2.0 * w_vlapl)

                        vmat[i] += numint._dot_ao_ao(mol, vg, ao[d], mask, shls_slice=None, ao_loc=None)

        if nset == 1:
            return nelec[0], excsum[0], vmat[0]
        else:
            return nelec, excsum, vmat
        
    def nr_uks(self, mol, grids, xc_code, dms, relativity=0, hermi=1,
               max_memory=2000, verbose=None):

        xctype = self._xc_type(xc_code)
        
        dma, dmb = _format_uks_dm(dms)
        nao = dma.shape[-1]
        
        make_rhoa, nset = self._gen_rho_evaluator(mol, dma, hermi, with_lapl=True, grids=grids)[:2]
        make_rhob = self._gen_rho_evaluator(mol, dmb, hermi, with_lapl=True, grids=grids)[0]

        if xctype == 'MGGA':
            ao_deriv = 2
        elif xctype == 'GGA':
            ao_deriv = 1
        else: 
            ao_deriv = 0

        nelec = numpy.zeros((2, nset))
        excsum = numpy.zeros(nset)
        vmat = numpy.zeros((2, nset, nao, nao))
        
        aow = None
        v1 = numpy.zeros_like(vmat) 
        pair_mask = mol.get_overlap_cond() < -numpy.log(self.cutoff)
        nbins = NBINS * 2 - int(NBINS * numpy.log(self.cutoff * 1e2) / numpy.log(grids.cutoff))

        for ao, mask, weight, coords in self.block_loop(mol, grids, nao, ao_deriv, max_memory=max_memory):
            for i in range(nset):
                rho_a = make_rhoa(i, ao, mask, xctype)
                rho_b = make_rhob(i, ao, mask, xctype)
                rho = (rho_a, rho_b)

                exc, vxc = self.eval_xc(xc_code, rho, spin=1, deriv=1)[:2]
                
                if len(vxc) == 4:
                    vrho, vsigma, vlapl, vtau = vxc
                else:
                    vrho, vsigma, vtau = vxc
                    vlapl = None

                den_a = rho_a[0] * weight if xctype != 'LDA' else rho_a * weight
                den_b = rho_b[0] * weight if xctype != 'LDA' else rho_b * weight
                nelec[0, i] += den_a.sum()
                nelec[1, i] += den_b.sum()
                excsum[i] += numpy.dot(den_a + den_b, exc)

                if vlapl is not None:
                    wva, wvb = numint._uks_mgga_wv0(rho, (vrho, vsigma, vlapl, vtau), weight)
                else:
                    wva, wvb = numint._uks_mgga_wv0(rho, (vrho, vsigma, vtau), weight)
                
                aow = _scale_ao_sparse(ao[:4], wva[:4], mask, ao_loc=mol.ao_loc_nr(), out=aow)
                _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, mol.ao_loc_nr(), hermi=0, out=vmat[0,i])

                if wva.shape[0] > 5:
                    _tau_dot_sparse(ao, ao, wva[5], nbins, mask, pair_mask, mol.ao_loc_nr(), out=v1[0,i])

                if vlapl is not None and numpy.linalg.norm(vlapl[:,0]) > 1e-9:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    
                    aow_lapl = _scale_ao(ao2, 0.5 * weight * vlapl[:,0])
                    mat_lapl = _dot_ao_ao(mol, ao[0], aow_lapl, mask, (0, mol.nbas), mol.ao_loc_nr())
                    vmat[0,i] += mat_lapl + mat_lapl.T.conj()
                    
                    for d in [1, 2, 3]:
                        vg = _scale_ao(ao[d], 0.5 * 2.0 * weight * vlapl[:,0]) 
                        mat_grad = _dot_ao_ao(mol, vg, ao[d], mask, (0, mol.nbas), mol.ao_loc_nr())
                        vmat[0,i] += mat_grad

                aow = _scale_ao_sparse(ao[:4], wvb[:4], mask, ao_loc=mol.ao_loc_nr(), out=aow)
                _dot_ao_ao_sparse(ao[0], aow, None, nbins, mask, pair_mask, mol.ao_loc_nr(), hermi=0, out=vmat[1,i])
                
                if wvb.shape[0] > 5:
                    _tau_dot_sparse(ao, ao, wvb[5], nbins, mask, pair_mask, mol.ao_loc_nr(), out=v1[1,i])

                if vlapl is not None and numpy.linalg.norm(vlapl[:,1]) > 1e-9:
                    XX, YY, ZZ = 4, 7, 9
                    ao2 = ao[XX] + ao[YY] + ao[ZZ]
                    
                    aow_lapl = _scale_ao(ao2, 0.5 * weight * vlapl[:,1])
                    mat_lapl = _dot_ao_ao(mol, ao[0], aow_lapl, mask, (0, mol.nbas), mol.ao_loc_nr())
                    vmat[1,i] += mat_lapl + mat_lapl.T.conj()

                    for d in [1, 2, 3]:
                        vg = _scale_ao(ao[d], 0.5 * 2.0 * weight * vlapl[:,1])
                        mat_grad = _dot_ao_ao(mol, vg, ao[d], mask, (0, mol.nbas), mol.ao_loc_nr())
                        vmat[1,i] += mat_grad

        vmat = lib.hermi_sum(vmat.reshape(-1,nao,nao), axes=(0,2,1)).reshape(2,nset,nao,nao)
        vmat += v1

        if isinstance(dma, numpy.ndarray) and dma.ndim == 2:
            vmat = vmat[:,0]
            nelec = nelec.reshape(2)
            excsum = excsum[0]
        
        dtype = numpy.result_type(dma, dmb)
        if vmat.dtype != dtype:
            vmat = numpy.asarray(vmat, dtype=dtype)
            
        return nelec, excsum, vmat

class RKS_with_Laplacian(rks.RKS):
    """
    A custom RKS class that uses the NumIntForcedLaplacian engine.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numint = NumIntWithLaplacian()

class UKS_with_Laplacian(uks.UKS):
    """
    A custom UKS class that uses the NumIntForcedLaplacian engine.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._numint = NumIntWithLaplacian()