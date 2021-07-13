# LIBRARIES
import xarray as xr
from importlib import reload
import numpy as np
import matplotlib.pyplot as plt
from cdo import Cdo
import os
import warnings
import sys
from datetime import datetime as dtime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from collections.abc import Iterable
from regionmask.defined_regions import srex as srex_regions
from scipy import stats
plt.rcParams['hatch.linewidth']=0.3

class Dataset(xr.Dataset):
    '''
    Wrapper for xarray.Dataset class in order to add user-defined functions

    '''

    def __add__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]+other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]+other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]+other
        return self

    def __radd__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other+self[other.name]
        else:
            for var in self:
                self[var]=other+self[var]
        return self

    def __sub__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]-other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]-other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]-other
        return self

    def __rsub__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other-self[other.name]
        else:
            for var in self:
                self[var]=other-self[var]
        return self

    def __mul__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]*other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]*other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]*other
        return self

    def __rmul__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other*self[other.name]
        else:
            for var in self:
                self[var]=other*self[var]
        return self

    def __truediv__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]/other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]/other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]/other
        return self

    def __rtruediv__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other/self[other.name]
        else:
            for var in self:
                self[var]=other/self[var]
        return self

    def __floordiv__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]//other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]//other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]//other
        return self

    def __rfloordiv__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other//self[other.name]
        else:
            for var in self:
                self[var]=other//self[var]
        return self

    def __pow__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]**other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]**other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]**other
        return self

    def __rpow__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other**self[other.name]
        else:
            for var in self:
                self[var]=other**self[var]
        return self

    def __mod__(self,other):
        self=self.copy()
        if check_xarray(other,'Dataset'):
            for var in self:
                if var in [v for v in other]: self[var]=self[var]%other[var]
                else: self[var]=self[var]
        elif check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            for var in self:
                if var == other.name: self[var]=self[var]%other
                else: self[var]=self[var]
        else:
            for var in self:
                self[var]=self[var]%other
        return self

    def __rmod__(self,other):
        self=self.copy()
        if check_xarray(other,'DataArray'):
            if other.name not in [v for v in self]:
                raise Exception('Impossible to compute operation. Data variable'+
                                ' names mismatch.')
            self=other%self[other.name]
        else:
            for var in self:
                self[var]=other%self[var]
        return self

    def __getitem__(self, key):
        """Access variables or coordinates this dataset as a
        :myfuncs:class:`~__main__.DataArray`.

        Indexing with a list of names will return a new ``Dataset`` object.
        """
        from xarray.core import utils

        if utils.is_dict_like(key):
            return DataArray(self.isel(**key))

        if utils.hashable(key):
            return DataArray(self._construct_dataarray(key))
        else:
            return Dataset(self._copy_listed(np.asarray(key)),attrs=self.attrs)

    def get_spatial_coords(self):
        lats=["latitude","latitude_0","lat"]
        lons=["longitude","longitude_0","lon"]
        count=0
        for l in lats:
            if l in self.dims:
                coords=[l]
                break
            count+=1
        if count == 3:
            coords=[None]
        count=0
        for l in lons:
            if l in self.dims:
                coords.append(l)
                break
            count+=1
        if count == 3:
            coords.append(None)
        return coords

    def plotall(self,
                outpath = None):
        '''
        Plots all the variables of the Dataset using DataArray.plotvar function.

        '''
        
        if not os.path.isdir(outpath):
            raise Exception("outpath needs to be the directory where all the plots will be saved")
        if "annual_mean" in self.attrs:
            op = "amean"
        elif "seasonal_cycle" in self.attrs:
            op = "seascyc"
        if "anomalies" in self.attrs:
            op = "{}.anom".format(op)

        for var_name in self:
            full_outpath = os.path.join(outpath,".".join((var_name,op,"png")))
            plt.figure()
            self[var_name].plotvar(outpath = full_outpath)

    def annual_mean(self,num=None,copy=True,update_attrs=True):
        return annual_mean(self,num=num,copy=copy,update_attrs=update_attrs)

    def annual_cycle(self,num=None,copy=True,update_attrs=True):
        return annual_cycle(self,num=num,copy=copy,update_attrs=update_attrs)

    def seasonal_cycle(self,copy=True,update_attrs=True):
            return seasonal_cycle(self,copy=copy,update_attrs=update_attrs)

    def anomalies(self,base=None,copy=True,update_attrs=True):
        return anomalies(self,x_base=base,copy=copy,update_attrs=update_attrs)

    def average(self, dim=None, weights=None,**kwargs):
        if not check_xarray(self, 'Dataset'):
            exception_xarray(type='Dataset')
        if 'keep_attrs' not in kwargs: kwargs.update({'keep_attrs':True})
        return self.apply(average, dim=dim, weights=weights,**kwargs)

    def latitude_mean(self,copy=True,update_attrs=True):
        return latitude_mean(self,copy=copy,update_attrs=update_attrs)

    def global_mean(self,copy=True,update_attrs=True):
        return global_mean(self,copy=copy,update_attrs=update_attrs)

    def rms(self,copy=True,update_attrs=True):
        return rms(self,copy=copy,update_attrs=update_attrs)

    def to_celsius(self,copy=True):
        def func(x):
            try: return x.to_celsius()
            except: return x
        if copy: self = self.copy()
        return self.apply(lambda x: func(x),keep_attrs=True)

    def group_by(self,time_group,copy=True,update_attrs=True):
        return group_by(self,time_group,copy=copy,update_attrs=update_attrs)

    def srex_mean(self,copy=True):
        mask=SREX_regions.mask()
        if copy: self=self.copy()
        new=self.groupby(mask).mean('stacked_lat_lon')
        new.coords['srex_abbrev'] = ('srex_region', srex_regions.abbrevs)
        new.coords['srex_name'] = ('srex_region', srex_regions.names)
        return new

    def seasonal_time_series(self,first_month_num=None,update_attrs=True):
        return seasonal_time_series(self,first_month_num=first_month_num,
                                    update_attrs=update_attrs)

class DataArray(xr.DataArray):
    '''
    Wrapper for xarray.DataArray class in order to add user-defined functions

    '''

    def __add__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)+other,attrs=attrs)

    def __radd__(self,other):
        attrs=self.attrs
        return DataArray(other+xr.DataArray(self),attrs=attrs)

    def __sub__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)-other,attrs=attrs)

    def __rsub__(self,other):
        attrs=self.attrs
        return DataArray(other-xr.DataArray(self),attrs=attrs)

    def __mul__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)*other,attrs=attrs)

    def __rmul__(self,other):
        attrs=self.attrs
        return DataArray(other*xr.DataArray(self),attrs=attrs)

    def __truediv__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)/other,attrs=attrs)

    def __rtruediv__(self,other):
        attrs=self.attrs
        return DataArray(other/xr.DataArray(self),attrs=attrs)

    def __floordiv__(self, other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)//other,attrs=attrs)

    def __floordiv__(self,other):
        attrs=self.attrs
        return DataArray(other//xr.DataArray(self),attrs=attrs)

    def __pow__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)**other,attrs=attrs)

    def __rpow__(self,other):
        attrs=self.attrs
        return DataArray(other**xr.DataArray(self),attrs=attrs)

    def __mod__(self,other):
        attrs=self.attrs
        return DataArray(xr.DataArray(self)%other,attrs=attrs)

    def __rmod__(self,other):
        attrs=self.attrs
        return DataArray(other%xr.DataArray(self),attrs=attrs)

    def get_spatial_coords(self):
        lats=["latitude","latitude_0","lat"]
        lons=["longitude","longitude_0","lon"]
        count=0
        for l in lats:
            if l in self.dims:
                coords=[l]
                break
            count+=1
        if count == 3:
            coords=[None]
        count=0
        for l in lons:
            if l in self.dims:
                coords.append(l)
                break
            count+=1
        if count == 3:
            coords.append(None)
        return coords

    def plotvar(self, projection = None,
                outpath = None,
                name = None,
                title = None,
                statistics='all',
                t_student=False,
                nlev=None,
                du= None,
                coast_kwargs = None,
                land_kwargs = None,
                save_kwargs = None,
                **contourf_kwargs):

        '''
        Plots GREB variables with associated color maps, scales and projections.

        '''

        def _get_var(x):
            keys = x.attrs.keys()
            name = x.name if x.name is not None else 'data'
            if 'long_name' in keys:
                title = x.attrs['long_name'] if x.attrs['long_name'] is not None else ''
            else:
                title = x.name if x.name is not None else ''
            if 'units' in keys:
                units = x.attrs['units'] if x.attrs['units'] is not None else ''
            else:
                units = ''
            if 'annual_mean' in keys:
                title = title + ' Annual Mean'
                name=name+'.amean'
            if 'seasonal_cycle' in keys:
                title = title + ' Seasonal Cycle'
                name=name+'.seascyc'
                cmap = cm.RdBu_r
            if 'anomalies' in keys:
                title = title + ' Anomalies'
                name=name+'.anom'
            return title,name,units

        param=self._get_param(nlev=nlev)
        self=param['self']
        if title is None: title = _get_var(self)[0]
        if name is None: name = _get_var(self)[1]
        if nlev is None: nlev=100
        units = _get_var(self)[2]

        if projection is None: projection = ccrs.Robinson()
        elif not projection: projection = ccrs.PlateCarree()
        if 'ax' not in contourf_kwargs:
            contourf_kwargs['ax'] = plt.axes(projection=projection)
        if 'cmap' not in contourf_kwargs:
            contourf_kwargs['cmap'] = param['cmap']
        if ('levels' not in contourf_kwargs) and ('norm' not in contourf_kwargs):    
            if param['levels'] is None:
                contourf_kwargs['levels'] = nlev
            else:
                contourf_kwargs['levels'] = param['levels']
        if ('add_colorbar' not in contourf_kwargs):contourf_kwargs['add_colorbar']=True
        if contourf_kwargs['add_colorbar']==True:
            cbar_kwargs = {'orientation':'horizontal', 'label':units}
            if 'cbar_kwargs' not in contourf_kwargs:
                contourf_kwargs["cbar_kwargs"] = cbar_kwargs
                contourf_kwargs["cbar_kwargs"]['ticks'] = param['cbticks']
            else:
                if 'orientation' not in contourf_kwargs['cbar_kwargs']: contourf_kwargs['cbar_kwargs']['orientation'] = 'horizontal'
                if 'label' not in contourf_kwargs['cbar_kwargs']: contourf_kwargs['cbar_kwargs']['label'] = units               
            if du is not None:
                umin=contourf_kwargs['levels'][0]
                umax=contourf_kwargs['levels'][-1]
                contourf_kwargs['cbar_kwargs'].update({"ticks":np.arange(umin,umax+du,du)})

        if land_kwargs is not None:
            land_kwargs = {'edgecolor':'face', 'facecolor':'black', **land_kwargs}
        else:
            land_kwargs = {'edgecolor':'face', 'facecolor':'black'}

        if coast_kwargs is not None:
            coast_kwargs = {**coast_kwargs}
        else:
            coast_kwargs = {}

        if save_kwargs is not None:
            save_kwargs = {'dpi':300, 'bbox_inches':'tight', **save_kwargs}
        else:
            save_kwargs = {'dpi':300, 'bbox_inches':'tight'}

        im=self._to_contiguous_lon().plot.contourf(transform=ccrs.PlateCarree(),
                                                    **contourf_kwargs)

        plt.gca().add_feature(cfeature.COASTLINE,**coast_kwargs)
        if (self.name == 'tocean'):
            plt.gca().add_feature(cfeature.NaturalEarthFeature('physical', 'land', '110m'),
                                  **land_kwargs)

        fmt = ".2f"
        fs = 12
        
        if isinstance(statistics,dict):
            if "fs" in statistics:
                fs = statistics["fs"]
            elif "fontsize" in statistics:
                fs = statistics["fontsize"]
            if "format" in statistics:
                fmt = statistics["format"]
            if "value" not in statistics: 
                raise Exception("Missing 'value' key in statistics. 'value' must be either 'all', 'gmean' or 'rms'.")
            else: statistics = statistics['value']

        if isinstance(statistics,str):
            if statistics == 'all':
                gm=self.global_mean().values
                rms=self.rms().values
                txt = (f'gmean = {gm:{fmt}}  |  rms = {rms:{fmt}}').format(gm,rms)
                plt.text(0.5,-0.05,txt,verticalalignment='top',horizontalalignment='center',
                        transform=plt.gca().transAxes,fontsize=fs, weight='bold')
            elif statistics == 'gmean':
                gm=self.global_mean().values
                txt = (f'gmean = {gm:{fmt}}')
                plt.text(0.5,-0.05,txt,verticalalignment='top',horizontalalignment='center',
                        transform=plt.gca().transAxes,fontsize=fs, weight='bold')
            elif statistics == 'rms':
                rms=self.rms().values
                txt = (f'rms = {rms:{fmt}}')
                plt.text(0.5,-0.05,txt,verticalalignment='top',horizontalalignment='center',
                        transform=plt.gca().transAxes,fontsize=fs, weight='bold')
            else: raise Exception("Invalid string for statistics. statistics must be either 'all', 'gmean' or 'rms'.")
        else: raise Exception("Invalid type for statistics. statistics must be either 'all', 'gmean' or 'rms'.")
      
        if isinstance(t_student,bool):
            if t_student:
                raise Exception('t_student must be False, or equal to either a dictionary or an '
                        'xarray.DataArray containing t-student distribution probabilities.')
        else:
            if check_xarray(t_student,"DataArray"):
                _check_shapes(t_student,self)
                t_student = {"p":t_student}
            elif isinstance(t_student,np.ndarray):
                t_student = {"p":xr.DataArray(data=t_student,dims = ["latitude","longitude"], coords=[UM.latitude,UM.longitude])}
            if isinstance(t_student,dict):
                if "p" not in t_student:
                    raise Exception('t_student must be contain "p" key, containing '
                        'an xarray.DataArray with t-student distribution '
                        'probabilities.\nTo obtain t_student distribution '
                        'probabilities, you can use the "t_student_probability" function.')
                elif isinstance(t_student["p"],np.ndarray):
                    t_student["p"] = xr.DataArray(data=t_student["p"],dims = ["latitude","longitude"], coords=[UM.latitude,UM.longitude])
                if "treshold" in t_student:
                    if t_student["treshold"] > 1:
                        raise Exception("Treshold must be <= 1")
                else:
                    t_student["treshold"]=0.05
                if "hatches" not in t_student:
                    t_student["hatches"]= '///'
            else:
                raise Exception('t_student must be either a dictionary or an '
                        'xarray.DataArray containing t-student distribution probabilities.')
            p=t_student["p"]
            a=t_student["treshold"]
            DataArray(p.where(p<a,0).where(p>=a,1))._to_contiguous_lon().plot.contourf(
                                                ax=plt.gca(),
                                                transform=ccrs.PlateCarree(),
                                                levels=np.linspace(0,1,3),
                                                hatches=['',t_student['hatches']],
                                                alpha=0,
                                                add_colorbar=False,
                                                )
        plt.title(title)
        if outpath is not None:
            plt.savefig(outpath, format = 'png',**save_kwargs)
            # plt.close()
        return im

    def _get_param(self,nlev=None):
        '''
        Function to set parameter for plotting

        '''

        cmap_tsurf=Colormaps.div_tsurf
        cmap_precip=Colormaps.div_precip
        keys=self.attrs.keys()
        name=self.name
        if nlev is None: nlev=100
        levels = None
        cbticks = None
        cmap = None
        # TATMOS
        if name == 'tatmos':
            cmap = cm.RdBu_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-2,2,nlev)
                cbticks = np.arange(-2,2+0.4,0.4)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(223,323,nlev)
                    cbticks = np.arange(223,323+10,10)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-20,20,nlev)
                    cbticks = np.arange(-20,20+4,4)
        # TSURF
        elif name == 'tsurf':
            cmap = cm.RdBu_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-2,2,nlev)
                cbticks = np.arange(-2,2+0.4,0.4)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(223,323,nlev)
                    cbticks = np.arange(223,323+10,10)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-20,20,nlev)
                    cbticks = np.arange(-20,20+4,4)
        # TOCEAN
        elif name == 'tocean':
            cmap = cm.RdBu_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-1,1,nlev)
                cbticks = np.arange(-1,1+0.2,0.2)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(273,303,nlev)
                    cbticks = np.arange(273,303+10,3)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-1,1,nlev)
                    cbticks = np.arange(-1,1+2e-1,2e-1)
        # PRECIP
        elif name == 'precip':
            cmap = cm.GnBu
            if 'anomalies' in keys:
                cmap = cmap_precip
                levels = np.linspace(-1,1,nlev)
                cbticks = np.arange(-1,1+0.2,0.2)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(0,9,nlev)
                    cbticks = np.arange(0,9+1,1)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_precip
                    levels = np.linspace(-6,6,nlev)
                    cbticks = np.arange(-6,6+1,1)
        # EVA
        elif name == 'eva':
            cmap = cm.GnBu
            if 'anomalies' in keys:
                cmap = cmap_precip
                levels = np.linspace(-1,1,nlev)
                cbticks = np.arange(-1,1+0.2,0.2)
            else:
                if 'annual_mean' in keys:
                    cmap = cm.Blues
                    levels = np.linspace(0,10,nlev)
                    cbticks = np.arange(0,10+1,1)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_precip
                    levels = np.linspace(-3,3,nlev)
                    cbticks = np.arange(-3,3+0.5,0.5)
        # QCRCL
        elif name == 'qcrcl':
            cmap = cm.RdBu_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-1,1,nlev)
                cbticks = np.arange(-1,1+0.2,0.2)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(-8,8,nlev)
                    cbticks = np.arange(-8,8+2,2)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-6,6,nlev)
                    cbticks = np.arange(-6,6+1,1)
        # VAPOR
        elif name == 'vapor':
            cmap = cm.RdBu_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-5e-3,5e-3,nlev)
                cbticks = np.arange(-5e-3,5e-3+1e-3,1e-3)
            else:
                if 'annual_mean' in keys:
                    cmap = cm.Blues
                    levels = np.linspace(0,0.02,nlev)
                    cbticks = np.arange(0,0.02+0.002,0.002)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-0.01,0.01,nlev)
                    cbticks = np.arange(-0.01,0.01+0.0012,0.002)
        # ICE
        elif name == 'ice':
            cmap = cm.Blues_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-1,1,nlev)
                cbticks = np.arange(-1,1+0.2,0.2)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(0,1,nlev)
                    cbticks = np.arange(0,1+0.1,0.1)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(0,1,nlev)
                    cbticks = np.arange(0,1+0.1,0.1)
        # SW
        elif name == 'sw':
            cmap = cm.YlOrRd
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-5,5,nlev)
                cbticks = np.arange(-5,5+1,1)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(50,350,nlev)
                    cbticks = np.arange(50,350+50,50)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-180,180,nlev)
                    cbticks = np.arange(-180,180+40,40)
        # CLOUD
        elif name == 'cloud':
            cmap = cm.Greys_r
            if 'anomalies' in keys:
                cmap = cmap_tsurf
                levels = np.linspace(-0.5,0.5,nlev)
                cbticks = np.arange(-0.5,0.5+0.1,0.1)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(0,1,nlev)
                    cbticks = np.arange(0,1+0.1,0.1)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-1,1,nlev)
                    cbticks = np.arange(-1,1+0.2,0.2)
        # SOLAR
        elif name == 'solar':
            if len(self.shape) == 1:
                self=GREB.def_DataArray(data=np.broadcast_to(self,[GREB.dx()]+list(self.shape)).transpose(),
                                            dims=('lat','lon'),
                                            attrs=self.attrs)
            cmap = cm.YlOrRd
            if 'anomalies' in keys:
                if 'annual_mean' in keys:
                    cmap = cm.hsv
                    levels = np.linspace(-8,0,nlev)
                    cbticks = np.arange(-8,1,1)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-5,5,nlev)
                    cbticks = np.arange(-5,5+1,1)
            else:
                if 'annual_mean' in keys:
                    levels = np.linspace(150,420,nlev)
                    cbticks = np.arange(150,420+40,40)
                elif 'seasonal_cycle' in keys:
                    cmap = cmap_tsurf
                    levels = np.linspace(-230,230,nlev)
                    cbticks = np.arange(-230,230+40,40)
        return {'self':self,'levels':levels,'cbticks':cbticks,'cmap':cmap}

    def plotlev(self,
                title = None,
                units = None,
                t_student=False,
                outpath = None,
                save_kwargs = None,
                **contourf_kwargs):

        if ("latitude" in self.dims) or ("latitude_0" in self.dims) or ("lat" in self.dims):
            core_dim="lat"
        elif  ("longitude" in self.dims) or ("longitude_0" in self.dims) or ("lon" in self.dims):
            core_dim="lon"
            lon = self.get_spatial_coords()[1]
            l=len(self[lon])
            self = self.roll({lon:int(l/2)})
            oldlon=self[lon]
            newlon=oldlon.where(oldlon < 180, oldlon-360)
            self=self.assign_coords({lon:newlon})._to_contiguous_lon()

        if "pressure" in self.dims:
            vertical_levs = "pressure"
        elif "model_level_number" in self.dims: 
            vertical_levs = "um_levs"
        else:
            vertical_levs = None

        ax = plt.axes() if 'ax' not in contourf_kwargs else contourf_kwargs.pop('ax')
        if ('add_colorbar' not in contourf_kwargs):contourf_kwargs['add_colorbar']=True
        if contourf_kwargs['add_colorbar']==True:
            if 'cbar_kwargs' not in contourf_kwargs: contourf_kwargs['cbar_kwargs'] = dict()
            if units is not None:
                contourf_kwargs['cbar_kwargs']['label']=units
        yscale = "log" if vertical_levs == "pressure" else "linear"   
        yincrease = False if vertical_levs == "um_levs" else True
            
        im=self.plot.contourf(ax=ax,
                    yincrease=yincrease,
                    yscale=yscale,
                    **contourf_kwargs,
                    )
        if isinstance(t_student,bool):
            if t_student:
                raise Exception('t_student must be False, or equal to either a dictionary or an '
                        'xarray.DataArray containing t-student distribution probabilities.')
        else:
            if check_xarray(t_student,"DataArray"):
                _check_shapes(t_student,self)
                t_student = {"p":t_student}
            elif isinstance(t_student,np.ndarray):
                t_student = {"p":xr.DataArray(data=t_student,dims = ["latitude","longitude"], coords=[UM.latitude,UM.longitude])}
            if isinstance(t_student,dict):
                if "p" not in t_student:
                    raise Exception('t_student must be contain "p" key, containing '
                        'an xarray.DataArray with t-student distribution '
                        'probabilities.\nTo obtain t_student distribution '
                        'probabilities, you can use the "t_student_probability" function.')
                elif isinstance(t_student["p"],np.ndarray):
                    t_student["p"] = xr.DataArray(data=t_student["p"],dims = ["latitude","longitude"], coords=[UM.latitude,UM.longitude])
                if "treshold" in t_student:
                    if t_student["treshold"] > 1:
                        raise Exception("Treshold must be <= 1")
                else:
                    t_student["treshold"]=0.05
                if "hatches" not in t_student:
                    t_student["hatches"]= '///'
            else:
                raise Exception('t_student must be either a dictionary or an '
                        'xarray.DataArray containing t-student distribution probabilities.')
            p=t_student["p"]
            a=t_student["treshold"]
            P=p.where(p<a,0).where(p>=a,1)
            if core_dim == "lon": 
                P = P.roll({lon:int(l/2)}).assign_coords({lon:newlon})._to_contiguous_lon()
            DataArray(P).plot.contourf(
                                        yincrease=False,
                                        yscale=yscale,
                                        levels=np.linspace(0,1,3),
                                        hatches=['',t_student['hatches']],
                                        alpha=0,
                                        add_colorbar=False,
                                        )
        plt.xlabel("")
        if core_dim == "lat":
            plt.xticks(ticks=np.arange(-90,90+30,30),
                       labels=["90S","60S","30S","0","30N","60N","90N"])
            plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
        elif core_dim == "lon":
            plt.xticks(ticks=np.arange(-180,180+60,60),
                       labels=["180W","120W","60W","0","60E","120E","180E"])
            plt.gca().xaxis.set_minor_locator(MultipleLocator(10))
        if vertical_levs=="pressure": 
            plt.yticks(ticks=[1000,800,600,400,200,50],labels=["1000","800","600","400","200","50"])
            plt.ylim([1000,50])
            # plt.ylabel("Pressure")
        elif vertical_levs=="um_levs": 
            plt.ylim([1,38])    
            plt.yticks(ticks=np.arange(1,38,5))
            plt.ylabel("Model level number")
        plt.tick_params(axis='y',which='minor',left=False,right=False)
        plt.tick_params(axis='y',which='major',left=True,right=True)
        plt.tick_params(axis='x',which='both',bottom=True,top=True)
        if title is None: title=self.name
        plt.title(title)
        if save_kwargs is not None:
            save_kwargs = {'dpi':300, 'bbox_inches':'tight', **save_kwargs}
        else:
            save_kwargs = {'dpi':300, 'bbox_inches':'tight'}
        if outpath is not None:
            plt.savefig(outpath, format = 'png',**save_kwargs)
        return im

    def _to_contiguous_lon(self):
        '''
        Function to close the longitude coord (for plotting)

        '''
        lon=self.get_spatial_coords()[1]
        if np.all(self[lon]>=0):
            if (0 in self[lon]) and (360 in self[lon]):
                return
            elif (0 in self[lon]):
                return xr.concat([self, self.sel({lon:0}).assign_coords({lon:360.})], dim=lon)
            elif (360 in self[lon]):
                return xr.concat([self.sel({lon:360}).assign_coords({lon:0.}), self], dim=lon)
            else:
                val=self.isel({lon:[0,-1]}).mean(lon)
                return xr.concat([val.assign_coords({lon:0.}),self,val.assign_coords({lon:360.})], dim=lon)
        else:
            if (-180 in self[lon]) and (180 in self[lon]):
                return
            elif (-180 in self[lon]):
                return xr.concat([self, self.sel({lon:-180}).assign_coords({lon:180.})], dim=lon)
            elif (180 in self[lon]):
                return xr.concat([self.sel({lon:180}).assign_coords({lon:-180.}), self], dim=lon)
            else:
                val=self.isel({lon:[0,-1]}).mean(lon)
                return xr.concat([val.assign_coords({lon:-180.}),self,val.assign_coords({lon:180.})], dim=lon)    

    def annual_mean(self,num=None,copy=True,update_attrs=True):
        return annual_mean(self,num=num,copy=copy,update_attrs=update_attrs)
    
    def annual_cycle(self,num=None,copy=True,update_attrs=True):
        return annual_cycle(self,num=num,copy=copy,update_attrs=update_attrs)
    
    def seasonal_cycle(self,copy=True,update_attrs=True):
        return seasonal_cycle(self,copy=copy,update_attrs=update_attrs)

    def anomalies(self,base=None,copy=True,update_attrs=True):
        return anomalies(self,x_base=base,copy=copy,update_attrs=update_attrs)

    def average(self, dim=None, weights=None,**kwargs):
        if 'keep_attrs' not in kwargs: kwargs.update({'keep_attrs':True})
        return average(self, dim=dim, weights=weights,**kwargs)
    
    def latitude_mean(self,copy=True,update_attrs=True):
        return latitude_mean(self,copy=copy,update_attrs=update_attrs)

    def global_mean(self,copy=True,update_attrs=True):
        return global_mean(self,copy=copy,update_attrs=update_attrs)

    def rms(self,copy=True,update_attrs=True):
        return rms(self,copy=copy,update_attrs=update_attrs)

    def to_celsius(self,copy=True):
        if copy: self = self.copy()
        if self.attrs['units'] == 'K':
            self.attrs['units'] = 'C'
            attrs=self.attrs
            if 'anomalies' in self.attrs: return self
            newarray=self-273.15
            newarray.attrs = attrs
            return newarray
        elif self.attrs['units'] == 'C':
            return self
        else:
            raise Exception('Cannot convert to Celsius.\n{} '.format(self.name)+
                            'does not have temperature units.')

    def group_by(self,time_group,copy=True,update_attrs=True):
        return group_by(self,time_group,copy=copy,update_attrs=update_attrs)

    def srex_mean(self,copy=True):
        mask=SREX_regions.mask()
        if copy: self=self.copy()
        new=self.groupby(mask).mean('stacked_lat_lon')
        new.coords['srex_abbrev'] = ('srex_region', srex_regions.abbrevs)
        new.coords['srex_name'] = ('srex_region', srex_regions.names)
        return new

    def seasonal_time_series(self,first_month_num=None,update_attrs=True):
        return seasonal_time_series(self,first_month_num=first_month_num,
                                    update_attrs=update_attrs)

    def t_student_probability(self,y,season=None):
        return t_student_probability(self,y,season=season)

    def to_mm_per_day(self,copy=True):  
        if copy: self = self.copy()
        return UM.to_mm_per_day(self)

class UM:
    longitude = np.array([  0.  ,   3.75,   7.5 ,  11.25,  15.  ,  18.75,  22.5 ,  26.25,
        30.  ,  33.75,  37.5 ,  41.25,  45.  ,  48.75,  52.5 ,  56.25,
        60.  ,  63.75,  67.5 ,  71.25,  75.  ,  78.75,  82.5 ,  86.25,
        90.  ,  93.75,  97.5 , 101.25, 105.  , 108.75, 112.5 , 116.25,
        120.  , 123.75, 127.5 , 131.25, 135.  , 138.75, 142.5 , 146.25,
        150.  , 153.75, 157.5 , 161.25, 165.  , 168.75, 172.5 , 176.25,
        180.  , 183.75, 187.5 , 191.25, 195.  , 198.75, 202.5 , 206.25,
        210.  , 213.75, 217.5 , 221.25, 225.  , 228.75, 232.5 , 236.25,
        240.  , 243.75, 247.5 , 251.25, 255.  , 258.75, 262.5 , 266.25,
        270.  , 273.75, 277.5 , 281.25, 285.  , 288.75, 292.5 , 296.25,
        300.  , 303.75, 307.5 , 311.25, 315.  , 318.75, 322.5 , 326.25,
        330.  , 333.75, 337.5 , 341.25, 345.  , 348.75, 352.5 , 356.25],
    dtype=np.float32)
        
    latitude = np.array([-90. , -87.5, -85. , -82.5, -80. , -77.5, -75. , -72.5, -70. ,
        -67.5, -65. , -62.5, -60. , -57.5, -55. , -52.5, -50. , -47.5,
        -45. , -42.5, -40. , -37.5, -35. , -32.5, -30. , -27.5, -25. ,
        -22.5, -20. , -17.5, -15. , -12.5, -10. ,  -7.5,  -5. ,  -2.5,
        0. ,   2.5,   5. ,   7.5,  10. ,  12.5,  15. ,  17.5,  20. ,
        22.5,  25. ,  27.5,  30. ,  32.5,  35. ,  37.5,  40. ,  42.5,
        45. ,  47.5,  50. ,  52.5,  55. ,  57.5,  60. ,  62.5,  65. ,
        67.5,  70. ,  72.5,  75. ,  77.5,  80. ,  82.5,  85. ,  87.5,
        90. ], dtype=np.float32)

    hybrid_height = np.float32([9.99820613861084, 49.99888229370117, 130.00022888183594, 
                                249.9983367919922, 410.00103759765625, 610.00048828125, 
                                850.0006103515625,  1130.00146484375, 1449.9990234375, 
                                1810.0010986328125, 2210.0, 2649.99951171875, 3129.999755859375, 
                                3650.000732421875, 4209.99853515625, 4810.0009765625, 5450.0, 
                                6129.99951171875, 6850.0, 7610.0009765625, 8409.9990234375,  
                                9250.0009765625, 10130.0, 11050.0, 12010.0009765625, 13010.001953125, 
                                14050.400390625, 15137.7197265625, 16284.9736328125,17506.96875, 
                                18820.8203125, 20246.599609375, 21808.13671875, 23542.18359375, 
                                25520.9609375, 27901.357421875, 31063.888671875, 36081.76171875])
    
    @staticmethod
    def months(n_char=2):
        if n_char == 2:
            return ["ja","fb","mr","ar","my","jn","jl","ag","sp","ot","nv","dc"]
        elif n_char == 3:
            return ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
        else:
            raise Exception("n_char must be either 2 or 3.")

    @staticmethod
    def streams():
        return ['a','b','c','d','e','f','g','h','i','j']

    @staticmethod
    def ppm2kgkg(x):
        m_air=0.0289644 #kg/mol
        m_CO2=0.0440095 #kg/mol
        coeff=1E-6*(m_CO2/m_air)
        return x*coeff

    @staticmethod
    def kgkg2ppm(x):
        m_air=0.0289644 #kg/mol
        m_CO2=0.0440095 #kg/mol
        coeff=1E-6*(m_CO2/m_air)
        return x/coeff
    
    @staticmethod
    def um_years_ref():
        return {10:"a",11:"b",12:"c",13:"d",14:"e",15:"f",16:"g",17:"h",18:"i",19:"j",20:"k",21:"l",22:"m",
                23:"n",24:"o",25:"p",26:"q",27:"r",28:"s",29:"t",30:"u",31:"v",32:"w",33:"x",34:"y",35:"z"}

    @staticmethod
    def to_um_filename_years(years):
        if isinstance(years,list):
            return ['{}{:02d}'.format(UM.um_years_ref()[int(y/100)],int(round((y/100 % 1)*100))) for y in years]
        else:
            return '{}{:02d}'.format(UM.um_years_ref()[int(years/100)],int(round((years/100 % 1)*100)))

    @staticmethod            
    def from_um_filename_years(um_years):
        def get_key_by_value(val):
            for v in UM.um_years_ref().items():
                if v[1] == val:
                    return  v[0]
        if isinstance(um_years,list):
            return [get_key_by_value(y[0])*100+int(y[-2:]) for y in um_years]
        else:
            return get_key_by_value(um_years[0])*100+int(um_years[-2:])    
            
    @staticmethod
    def land_mask_file():
        return "/g/data3/w48/dm5220/ancil/land_mask/land_mask.nc"

    @staticmethod
    def add_evaporation(x):
        '''
        Function to add the variable "evaporation" to the Dataset.
        This variable is computed by adding the 2 variables of the dataset "evaporation_flux_from_open_sea" and "evaporation_from_soil_surface".

        Arguments
        ----------
        x : Dataset objects
        Dataset to add the "evaporation" variable to.

        Returns
        ----------
        xarray.Dataset

        New Dataset containing the "evaporation" variable.
        '''
    
        variables = ["evaporation_flux_from_open_sea","evaporation_from_soil_surface"]
        for v in variables:
            if v not in x.variables: raise Exception('Current Dataset doesn"t include the variable "{}".'.format(v))
        return x.assign(evaporation=x[variables[0]].where(x[variables[0]] <= 100,0) + 
                        x[variables[1]].where(x[variables[1]] <= 100,0))

    @staticmethod
    def rename_m01s09i231(x):
        '''
        Function to rename the variable m01s09i231 of UM model to combined_cloud_amount.
        '''
        return x.rename_vars({"m01s09i231":"combined_cloud_amount"})

    @staticmethod
    def split_w_wind(x):
        '''
        Function to split w component of winds into w_up and w_down
        '''
        d=x["upward_air_velocity"]
        d_up=d.where(d>0,0)
        d_down=-d.where(d<0,0)
        d_overtuning=d_down-d_up
        return x.assign({"omega_up":d_up,"omega_down":d_down,"omega_overtuning":d_overtuning})

    @staticmethod
    def to_mm_per_day(x):
        '''
        Function to convert data with units from "kg m-2 s-1" to "mm day-1"
        '''
        alpha = 86400
        if 'units' not in x.attrs:
            x.attrs['units']="TEMP_UNITS"
        if (x.attrs['units'] in ["kg m-2 s-1","kg/m^2/s"]):
            x.attrs['units']="mm/day"
            return x*alpha
        elif (x.name in ["precipitation_flux","evaporation_flux_from_open_sea",
                        "evaporation_from_soil_surface","evaporation"]):
            if x.attrs['units'] not in ["mm/day","mm day-1","mm d-1"]:
                x.attrs['units']="mm/day"
                return x*alpha
            else: return x
        else:
            raise Exception("Data units not understood")
            
class GREB:
    
    def __init__(self):
        pass

    @staticmethod 
    def lat():
        return  np.arange(-88.125,88.125+3.75,3.75)

    @staticmethod 
    def lon():
        return  np.arange(0,360,3.75)

    @staticmethod 
    def t():
        return np.arange(np.datetime64('2000-01-01T00:00:00.000000000'),
                         np.datetime64('2000-12-30T12:00:00.000000000')+np.timedelta64(12,'h'),
                         np.timedelta64(12,'h'))

    @staticmethod
    def dx():
        return len(GREB.lon())

    @staticmethod
    def dy():
        return len(GREB.lat())

    @staticmethod
    def dt():
        return len(GREB.t())

    @staticmethod
    def greb_folder():
        return "/Users/dmar0022/university/phd/greb-official"

    @staticmethod
    def figures_folder():
        return GREB.greb_folder()+'/figures'

    @staticmethod
    def output_folder():
        return GREB.greb_folder()+'/output'

    @staticmethod
    def input_folder():
        return GREB.greb_folder()+'/input'

    @staticmethod
    def scenario_2xCO2(years_of_simulation=50):
        return GREB.output_folder()+'/scenario.exp-20.2xCO2_{}yrs'.format(years_of_simulation)

    def cloud_def_file():
        return GREB.greb_folder()+'/input/isccp.cloud_cover.clim'

    @staticmethod
    def solar_def_file(tridimensional=True):
        if tridimensional:
            return GREB.greb_folder()+'/input/solar_radiation.clim_3D'
        else:
            return GREB.greb_folder()+'/input/solar_radiation.clim'

    @staticmethod
    def cloud_folder():
        return GREB.greb_folder()+'/artificial_clouds'

    @staticmethod
    def solar_folder():
        return GREB.greb_folder()+'/artificial_solar_radiation'

    @staticmethod
    def cloud_figures_folder():
        return GREB.cloud_folder()+'/art_clouds_figures'

    @staticmethod
    def control_def_file():
        return GREB.output_folder()+'/control.default'

    @staticmethod
    def to_greb_grid(x, method='cubic'):
        '''
        Regrid data to GREB lat/lon grid.

        Arguments
        ----------
        x : xarray.DataArray or xarray.Dataset
            Data to be regridded.

        Parameters
        ----------
        method: str
            Method for the interpolation.
            Can be chosen between: 'linear' (default), 'nearest', 'zero',
                                    'slinear', 'quadratic', 'cubic'.

        Returns
        ----------
        New xarray.Dataset or xarray.DataArray regridded into GREB lat/lon grid.

        '''

        grebfile=GREB.from_binary(GREB.control_def_file()).mean('time')
        return x.interp_like(grebfile,method=method)

    def def_DataArray(data=None,dims=None,coords=None,name=None,attrs=None):
        if dims is None:
            dims=('time','lat','lon')
        elif isinstance(dims,str):
            dims = [dims]
        if coords is None:
            if 'time' in dims: time = GREB.t()
            if 'lat' in dims: lat = GREB.lat()
            if 'lon' in dims: lon = GREB.lon()
        scope = locals()
        if name is not None:
            if attrs is None:
                attrs={'long_name':name}
            if 'long_name' not in attrs:
                attrs.update({'long_name':name})
        if coords is None:coords = dict((key,eval(key,scope)) for key in dims)
        if data is None: data = np.zeros([len(val) for val in coords.values()])
        return DataArray(data,name=name,dims=dims,coords=coords,attrs=attrs)

    @staticmethod
    def to_greb_indexes(lat,lon):
        '''
        Convert lat/lon from degrees to GREB indexes

        Arguments
        ----------
        lat : float or array of floats
            latitude point or array of latitude points.
        lon : float or array of floats
            longitude point or array of longitude points.
            lat and lon must be the same lenght

        Returns
        ----------
        GREB model Indexes corrisponding to the lat/lon couples.

        '''

        lat = np.array(lat)
        lon = np.array(lon)
        if np.any(lat<-90) or np.any(lat>90):
            raise ValueError('lat value must be between -90 and 90 degrees')
        lon=np.array(lon)
        if np.any(lon<0) or np.any(lon>360):
            raise ValueError('lon value must be between 0 and 360 degrees')
        lat_def = GREB.lat().tolist()
        lon_def = GREB.lon().tolist()
        i = [lat_def.index(min(lat_def, key=lambda x:abs(x-la))) for la in lat] \
            if len(lat.shape) == 1 else lat_def.index(min(lat_def, key=lambda x:abs(x-lat)))
        j = [lon_def.index(min(lon_def, key=lambda x:abs(x-lo))) for lo in lon] \
            if len(lon.shape) == 1 else lon_def.index(min(lon_def, key=lambda x:abs(x-lon)))
        return i,j

    @staticmethod
    def to_shape_for_bin(data,type=None,tridimensional=True):
        if not isinstance(tridimensional,bool):
            raise Exception("tridimensional must be either True or False")
        if type == 'cloud': tridimensional=True
        if tridimensional:
            def_sh=(GREB.dt(),GREB.dy(),GREB.dx())
        else:
            def_sh=(GREB.dt(),GREB.dy(),1)
        sh=data.shape
        if len(sh) == 2:
            data=np.expand_dims(data,axis=2)
            sh=data.shape
        elif len(sh) != 3: 
            raise Exception(f'data must be 3D, in the form {GREB.dt()}x{GREB.dy()}x{GREB.dx()} or {GREB.dt()}x{GREB.dy()}x1')
        if ((sh[0] not in def_sh) or (sh[1] not in def_sh) or (sh[2] not in def_sh)):
            raise Exception(f'data must be 3D, in the form {GREB.dt()}x{GREB.dy()}x{GREB.dx()} or {GREB.dt()}x{GREB.dy()}x1')
        if sh != def_sh:
            indt=sh.index(def_sh[0])
            indy=sh.index(def_sh[1])
            indx=sh.index(def_sh[2])
            data = data.transpose(indt,indy,indx)
        return data

    @staticmethod
    def get_years_of_simulation(sc_filename):
        '''
        Gets the number of years of simulations for the specified scenario file.

        Arguments
        ----------
        sc_filename : str
            Path to the scenario file

        Returns
        ----------
        int
            Number of years of simulations

        '''

        if 'yrs' not in sc_filename.split('_')[-1]:
            raise Exception('Could not understand the number of years of simulation.'+
                            '\nNumber of years of simulation N must be at the end of'+
                            ' sc_filename in the form "_Nyrs"')
        else:
            return (sc_filename.split('_')[-1])[:-3]

    @staticmethod
    def get_art_forcing_filename(sc_filename,forcing=None,output_path=None):
        '''
        Gets the artificial cloud filename used to obtain the scenario filename in input.

        Arguments
        ----------
        sc_filename : str
            Path to the scenario file

        Parameters
        ----------
        forcing : str
            Type of forcing to retrieve. Can be set to "cloud" or "solar".
            If not set, the forcing type will be tried to be understood from the
            scenario filename.
        output_path : str
            Path of the folder where to search for the artificial forcing file.

        Returns
        ----------
        str
            Path to the artificial cloud

        '''
        # Remove extension
        sc_filename = rmext(os.path.split(sc_filename)[1])
        # Remove years of simulation
        if sc_filename.endswith('yrs'):
            sc_filename = '_'.join(sc_filename.split('_')[:-1])
        # Get experiment number
        exp=sc_filename.split('.')[1]
        # Control run or other (non-geoengineering) experiments
        if exp not in ['exp-{}'.format(n) for n in ['930','931','932','933']]:
            if forcing in ['cloud','solar']:
                return eval('GREB.{}_def_file()'.format(forcing))
            else:
                raise Exception('Specify forcing to be either "cloud" or "solar".')
        else:
            # Get forcing
            if forcing is None:
                if exp == 'exp-930':
                    forcing = 'cloud'
                else:
                    forcing = 'solar'
            elif forcing not in ['cloud','solar']:
                raise Exception('Invalid forcing name "{}".\nForcing must be either "cloud" or "solar".'.format(forcing))
            # Get artificial forcing directory
            if output_path is None:
                output_path = eval('GREB.{}_folder()'.format(forcing))
            # Get artificial path
            if forcing == 'cloud':
                if exp == 'exp-930':
                    forcing_name='cld.artificial'
                    name=sc_filename[sc_filename.index(forcing_name):]
                    return os.path.join(output_path, name)
                else:
                    return eval('GREB.{}_def_file()'.format(forcing))
            elif forcing == 'solar':
                if exp == 'exp-930':
                    return eval('GREB.{}_def_file()'.format(forcing))
                else:
                    forcing_name='sw.artificial'
                    name=sc_filename[sc_filename.index(forcing_name):]
                    return os.path.join(output_path, name)

    @staticmethod
    def get_scenario_filename(forcing_filename,years_of_simulation=50,input_path=None):
        '''
        Gets the scenario filename from either an artifial cloud or solar forcing
        filename.

        Arguments
        ----------
        forcing_filename : str
            Path to the forcing file

        Parameters
        ----------
        years_of_simulation : int
            Number of years for which the forcing simulation has been run

        Returns
        ----------
        str
            Path to the output scenario

        '''

        txt1='cld.artificial'
        txt2='sw.artificial'
        forcing_filename = rmext(forcing_filename)
        forcing_filename_ = os.path.split(forcing_filename)[1]
        if txt1 in forcing_filename_:
            sc_name = 'scenario.exp-930.geoeng.2xCO2.'+forcing_filename_+'_{}yrs'.format(years_of_simulation)
        elif txt2 in forcing_filename_:
            sc_name = 'scenario.exp-931.geoeng.2xCO2.'+forcing_filename_+'_{}yrs'.format(years_of_simulation)
        elif (forcing_filename == GREB.cloud_def_file()) or (forcing_filename == GREB.solar_def_file()):
            sc_name = 'scenario.exp-20.2xCO2'+'_{}yrs'.format(years_of_simulation)
        else:
            raise Exception('The forcing file must contain either "cld.artificial" or "sw.artificial"')
        if input_path is None: input_path = GREB.output_folder()
        return os.path.join(input_path,sc_name)

    @staticmethod
    def days_each_month():
        return np.array([31,28,31,30,31,30,31,31,30,31,30,31])

    @staticmethod
    def land_ocean_mask():
        '''
        A land/ocean mask built from the GREB model topography input.
        True = Land
        False = Ocean

        '''
        mask=GREB.from_binary(GREB.input_folder()+'/global.topography.bin',parse=False).squeeze().topo
        mask.data=np.where(mask<=0,False,True)
        return mask
    
    @staticmethod
    def get_exp_name(exp_num):
        if isinstance(exp_num,str):
            exp_num = int(exp_num)
        elif not isinstance(exp_num,int):
            raise Exception("exp_num must be either an integer or a string")
        n1 = "exp-{}".format(exp_num)
        if exp_num in (930,931):
            n2 = "geoeng.2xCO2"
        elif exp_num == 932:
            n2 = "geoeng.4xCO2"
        elif exp_num == 933:
            n2 = "geoeng.control-fixed.tsurf.4xCO2"
        else:
            raise Exception("exp_num '{}' not supported".format(exp_num))
        return ".".join((n1,n2))

    @staticmethod
    def from_binary(filename,time_group=None,parse=True,use_netCDF=False):
        """
        Read binary file into an xarray.Dataset object.

        Arguments
        ----------
        filename : str
            Path to the ".bin" file to open

        Parameters
        ----------
        time_group : str
            Time grouping method to be chosen between: '12h','day','month','year','season'.
            If chosen, the retrieved data belonging to the same time_group will be
            averaged.
            If "time_group" is smaller than the data time-resolution, a spline
            interpolation is performed.
        parse: Bool
            Set to True (default) if you want the output to be parsed with the custom
            "parse_greb_var" function, otherwise set to False.

        Returns
        ----------
        xarray.Dataset
            Dataset containing all the variables in the binary file.

        """
        if time_group not in ['12h','day','month','year','season',None]:
            raise Exception('time_group must be one of the following:\n'+\
                            '"12h","day","month","year","season"')
        if use_netCDF:
            filename=rmext(filename)
            bin2netCDF(filename)
            data=xr.open_dataset(filename+".nc")
        else:
            from xgrads import open_CtlDataset
            filename = rmext(filename)+".ctl"
            data=open_CtlDataset(filename)
        
        attrs=data.attrs
        if parse: data = parse_greb_var(data)
        if 'time' in data.coords and data.time.shape[0] > 1:
            t_res = np.timedelta64(data.time.values[1]-data.time.values[0],'D').item().total_seconds()/(3600*12)
            if t_res > 62:
                raise Exception('Impossible to group data by "{}".\n'.format(time_group)+
                                'Could not understand the time resolution of the data.')
        if time_group is not None:
            return data.group_by(time_group,copy=False)
        else:
            return Dataset(data,attrs=attrs)

    @staticmethod
    def create_bin_ctl(path,vars,tridimensional=True):
        '''
        Creates '.bin' and '.ctl' file from Dictionary.

        Arguments
        ----------
        path : str
            complete path for the new '.bin' and '.ctl' files.
        vars : dictionary
            vars must be in the form {"namevar1":var1,"namevar2":var2,...,"namevarN":varN}
            var1,var2,...varN must have the same shape

        Returns
        ----------
        -
            Creates '.bin' and '.ctl' file from Dictionary.

        '''

        def _create_bin(path,vars):
            path = rmext(path)
            with open(path+'.bin','wb') as f:
                for v in vars:
                    f.write(v)

        def _create_ctl(path, varnames = None, xdef = None,
                    ydef = None, zdef = 1, tdef = None):
            path = rmext(path)
            if not isinstance(varnames,list): varnames = [varnames]
            nvars = len(varnames)
            name = os.path.split(path)[1]+'.bin'
            with open(path+'.ctl','w+') as f:
                f.write('dset ^{}\n'.format(name))
                f.write('undef 9.e27\n')
                f.write('xdef  {} linear 0 3.75\n'.format(xdef))
                f.write('ydef  {} linear -88.125 3.75\n'.format(ydef))
                f.write('zdef   {} linear 1 1\n'.format(zdef))
                f.write('tdef {} linear 00:00Z1jan2000  12hr\n'.format(tdef))
                f.write('vars {}\n'.format(nvars))
                for v in varnames:
                    f.write('{0}  1 0 {0}\n'.format(v))
                f.write('endvars\n')

        if not isinstance(vars,dict):
            raise Exception('vars must be a Dictionary type in the form: ' +
                            '{"namevar1":var1,"namevar2":var2,...,"namevarN":varN}')
        varnames = list(vars.keys())
        nvars = len(varnames)
        varvals = list(vars.values())
        varvals=[v.values if check_xarray(v) else v for v in varvals]
        l=[v.shape for v in varvals]
        if not ( l.count(l[0]) == len(l) ):
            raise Exception('var1,var2,...,varN must be of the same size')
        varvals = [GREB.to_shape_for_bin(v,type=n,tridimensional=tridimensional) for v,n in zip(varvals,varnames)]
        varvals=[np.float32(v.copy(order='C')) for v in varvals]
        tdef,ydef,xdef = varvals[0].shape
        # WRITE CTL FILE
        _create_ctl(path, varnames = varnames, xdef=xdef,ydef=ydef,tdef=tdef)
        # WRITE BIN FILE
        _create_bin(path,vars = varvals)

    @staticmethod
    def create_clouds(time = None, longitude = None, latitude = None, value = 1,
                    cloud_base = None, outpath = None):
        '''
        Create an artificial cloud matrix file from scratch or by modifying an
        existent cloud matrix.

        Parameters
        ----------
        time : array of str
        Datestr for the extent of the 'time' coordinate, in the form
        [time_min, time_max].
        Datestr format is '%y-%m-%d' (e.g. '2000-03-04' is 4th March 2000).

        longitude : array of float
        Extent of the 'lon' coordinate, in the form [lon_min, lon_max].
        Format: degrees -> 0° to 360°.

        latitude : array of float
        Extent of the 'lat' coordinate, in the form [lat_min, lat_max].
        Format: S-N degrees -> -90° to 90°.

        value : DataArray, np.ndarray, float or callable
        Cloud value to be assigned to the dimensions specified in time, latitude
        and longitude.
        If callable, equals the function to be applied element-wise to the
        "cloud_base" (e.g. "lambda x: x*1.1" means cloud_base scaled by 1.1).

        cloud_base : xarray.DataArray, np.ndarray or str
        Array of the cloud to be used as a reference for the creation of the
        new matrix or full path to the file ('.bin' and '.ctl' files).

        outpath : str
        Full path where the new cloud file ('.bin' and '.ctl' files)
        is saved.
        If not provided, the following default path is chosen:
        '/Users/dmar0022/university/phd/GREB-official/artificial_clouds/cld.artificial'

        Returns
        ----------
        -
        Creates artificial cloud file ('.bin' and '.ctl' files).

        '''
        if cloud_base is not None:
            if isinstance(cloud_base,str):
                data=GREB.from_binary(cloud_base).cloud
            elif isinstance(cloud_base,np.ndarray):
                data = GREB.def_DataArray(data=cloud_base,name='cloud')
            elif isinstance(cloud_base,xr.DataArray):
                data = DataArray(cloud_base,attrs=cloud_base.attrs)
            else:
                raise Exception('"cloud_base" must be a xarray.DataArray, numpy.ndarray,'+
                                ' or a valid path to the cloud file.')
        else:
            if callable(value):
                data = GREB.from_binary(GREB.cloud_def_file()).cloud
            else:            
                data = GREB.def_DataArray(name='cloud')

        mask=True
        # Check coordinates and constrain them
        # TIME
        if time is not None:
            t_exc = '"time" must be a np.datetime64, datestring or list of np.datetime64 istances or datestrings, in the form [time_min,time_max].\n'+\
                    'Datestring must be in the format "%y-%m-%d".'
            if isinstance(time,Iterable):
                if not(isinstance(time,str)) and (len(time) > 2):
                    raise Exception(t_exc)
                if isinstance(time,str):
                    time = np.datetime64(time)
                    mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
                else:
                    time = [np.datetime64(t) for t in time]
                    if time[0]==time[1]:
                        time = time[0]
                        mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
                    elif time[1]>time[0]:
                        mask = mask&(data.coords['time']>=time[0])&(data.coords['time']<=time[1])
                    else:
                        mask = mask&((data.coords['time']>=time[0])|(data.coords['time']<=time[1]))
            elif isinstance(time,np.datetime64):
                mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
            else:
                raise Exception(t_exc)
        else:
            mask=mask&(GREB.def_DataArray(np.full(GREB.dt(),True),dims='time'))

        # LONGITUDE
        if longitude is not None:
            lon_exc = '"longitude" must be a number or in the form [lon_min,lon_max]'
            if isinstance(longitude,Iterable):
                if (isinstance(longitude,str)) or (len(longitude) > 2):
                    raise Exception(lon_exc)
                elif longitude[1]==longitude[0]:
                    longitude = np.array(longitude[0])
                    if (longitude<0) or (longitude>360):
                        raise ValueError('"longitude" must be in the range [0÷360]')
                    else:
                        mask = mask&(data.coords['lon']==data.coords['lon'][np.argmin(abs(data.coords['lon']-longitude))])
                elif longitude[1]>longitude[0]:
                    mask = mask&((data.coords['lon']>=longitude[0])&(data.coords['lon']<=longitude[1]))
                else:
                    mask = mask&((data.coords['lon']>=longitude[0])|(data.coords['lon']<=longitude[1]))
            elif (isinstance(longitude,float) or isinstance(longitude,int)):
                longitude=np.array(longitude)
                if np.any([longitude<0,longitude>360]):
                    raise ValueError('"longitude" must be in the range [0÷360]')
                else:
                    mask = mask&(data.coords['lon']==data.coords['lon'][np.argmin(abs(data.coords['lon']-longitude))])
            else:
                raise Exception(lon_exc)
        else:
            mask=mask&(GREB.def_DataArray(np.full(GREB.dx(),True),dims='lon'))

        # LATITUDE
        if latitude is not None:
            lat_exc = '"latitude" must be a number or in the form [lat_min,lat_max]'
            if isinstance(latitude,Iterable):
                if (isinstance(latitude,str)) or (len(latitude) > 2):
                    raise Exception(lat_exc)
                elif latitude[1]==latitude[0]:
                    latitude = np.array(latitude[0])
                    if np.any([latitude<-90,latitude>90]):
                        raise ValueError('"latitude" must be in the range [-90÷90]')
                    else:
                        mask = mask&data.coords['lat']==data.coords['lat'][np.argmin(abs(data.coords['lat']-latitude))]
                elif latitude[1]>latitude[0]:
                    mask = mask&(data.coords['lat']>=latitude[0])&(data.coords['lat']<=latitude[1])
                else:
                    mask = mask&((data.coords['lat']>=latitude[0])|(data.coords['lat']<=latitude[1]))
            elif (isinstance(latitude,float) or isinstance(latitude,int)):
                if (latitude<-90) or (latitude>90):
                    raise ValueError('"latitude" must be in the range [0÷360]')
                else:
                    mask = mask&(data.coords['lat']==data.coords['lat'][np.argmin(abs(data.coords['lat']-latitude))])
            else:
                raise Exception(lat_exc)
        else:
            mask=mask&(GREB.def_DataArray(np.full(GREB.dy(),True),dims='lat'))

        # Change values
        if (isinstance(value,float) or isinstance(value,int) or isinstance(value,np.ndarray)):
            data=data.where(~mask,value)
        elif isinstance(value,xr.DataArray):
            data=data.where(~mask,value.values)
        elif callable(value):
            data=data.where(~mask,value(data))
        else:
            raise Exception('"value" must be a number, numpy.ndarray, xarray.DataArray or function to apply to the "cloud_base" (e.g. "lambda x: x*1.1")')
        # Correct value above 1 or below 0
        data=data.where(data<=1,1)
        data=data.where(data>=0,0)
        # Write .bin and .ctl files
        vars = {'cloud':data.values}
        if outpath is None:
            outpath=GREB.cloud_folder()+'/cld.artificial.ctl'
        GREB.create_bin_ctl(outpath,vars)

    @staticmethod
    def create_solar(time = None, longitude = None, latitude = None, value = 1,
                    solar_base = None, tridimensional = True, outpath = None):
        '''
        Create an artificial solar matrix file from scratch or by modifying an
        existent solar matrix.

        Parameters
        ----------
        time : array of str
        Datestr for the extent of the 'time' coordinate, in the form
        [time_min, time_max].
        Datestr format is '%y-%m-%d' (e.g. '2000-03-04' is 4th March 2000).

        latitude : array of float
        Extent of the 'lat' coordinate, in the form [lat_min, lat_max].
        Format: S-N degrees -> -90° to 90°.

        longitude : array of float
        Extent of the 'lon' coordinate, in the form [lon_min, lon_max].
        Format: degrees -> 0° to 360°.

        value : float or callable
        Solar value to be assigned to the dimensions specified in time and
        latitude.
        If callable, equals the function to be applied element-wise to the
        "solar_base" (e.g. "lambda x: x*1.1" means solar_base scaled by 1.1).

        solar_base : np.ndarray or str
        Array of the solar to be used as a reference for the creation of the
        new matrix or full path to the file ('.bin' and '.ctl' files).

        outpath : str
        Full path where the new solar file ('.bin' and '.ctl' files)
        is saved.
        If not provided, the following default path is chosen:
        '/Users/dmar0022/university/phd/GREB-official/artificial_solar_radiation/sw.artificial'

        Returns
        ----------
        -
        Creates artificial solar file ('.bin' and '.ctl' files).

        '''
        if (not tridimensional) and (longitude is not None):
            raise Exception("Requested change in longitude when data not set to tridimensional. Set 'tridimensional'=True if you want to change longitude")
        if solar_base is not None:
            if isinstance(solar_base,str):
                data=GREB.from_binary(solar_base).solar
            elif isinstance(solar_base,np.ndarray):
                data = GREB.def_DataArray(data=solar_base,name='solar')
            elif isinstance(solar_base,xr.DataArray):
                data = DataArray(solar_base,attrs=solar_base.attrs)
            else:
                raise Exception('"solar_base" must be a xarray.DataArray, numpy.ndarray,'+
                                ' or a valid path to the solar file.')
        else:
            if callable(value):
                data = GREB.from_binary(GREB.solar_def_file(tridimensional=tridimensional)).solar
            else:
                if tridimensional:         
                    data = GREB.def_DataArray(name='solar',dims=('time','lat','lon'))
                else:
                    data = GREB.def_DataArray(name='solar',dims=('time','lat'))

        mask=True
        # Check coordinates and constrain them
        # TIME
        if time is not None:
            t_exc = '"time" must be a np.datetime64, datestring or list of np.datetime64 istances or datestrings, in the form [time_min,time_max].\n'+\
                    'Datestring must be in the format "%y-%m-%d".'
            if isinstance(time,Iterable):
                if not(isinstance(time,str)) and (len(time) > 2):
                    raise Exception(t_exc)
                if isinstance(time,str):
                    time = np.datetime64(time)
                    mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
                else:
                    time = [np.datetime64(t) for t in time]
                    if time[0]==time[1]:
                        time = time[0]
                        mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
                    elif time[1]>time[0]:
                        mask = mask&(data.coords['time']>=time[0])&(data.coords['time']<=time[1])
                    else:
                        mask = mask&((data.coords['time']>=time[0])|(data.coords['time']<=time[1]))
            elif isinstance(time,np.datetime64):
                mask = mask&data.coords['time']==data.coords['time'][np.argmin(abs(data.coords['time']-time))]
            else:
                raise Exception(t_exc)
        else:
            mask=mask&(GREB.def_DataArray(np.full(GREB.dt(),True),dims='time'))

        # LATITUDE
        if latitude is not None:
            lat_exc = '"latitude" must be a number or in the form [lat_min,lat_max]'
            if isinstance(latitude,Iterable):
                if (isinstance(latitude,str)) or (len(latitude) > 2):
                    raise Exception(lat_exc)
                elif latitude[1]==latitude[0]:
                    latitude = np.array(latitude[0])
                    if np.any([latitude<-90,latitude>90]):
                        raise ValueError('"latitude" must be in the range [-90÷90]')
                    else:
                        mask = mask&data.coords['lat']==data.coords['lat'][np.argmin(abs(data.coords['lat']-latitude))]
                elif latitude[1]>latitude[0]:
                    mask = mask&(data.coords['lat']>=latitude[0])&(data.coords['lat']<=latitude[1])
                else:
                    mask = mask&((data.coords['lat']>=latitude[0])|(data.coords['lat']<=latitude[1]))
            elif (isinstance(latitude,float) or isinstance(latitude,int)):
                if (latitude<-90) or (latitude>90):
                    raise ValueError('"latitude" must be in the range [0÷360]')
                else:
                    mask = mask&(data.coords['lat']==data.coords['lat'][np.argmin(abs(data.coords['lat']-latitude))])
            else:
                raise Exception(lat_exc)
        else:
            mask=mask&(GREB.def_DataArray(np.full(GREB.dy(),True),dims='lat'))

        if tridimensional:
            # LONGITUDE
            if longitude is not None:
                lon_exc = '"longitude" must be a number or in the form [lon_min,lon_max]'
                if isinstance(longitude,Iterable):
                    if (isinstance(longitude,str)) or (len(longitude) > 2):
                        raise Exception(lon_exc)
                    elif longitude[1]==longitude[0]:
                        longitude = np.array(longitude[0])
                        if (longitude<0) or (longitude>360):
                            raise ValueError('"longitude" must be in the range [0÷360]')
                        else:
                            mask = mask&(data.coords['lon']==data.coords['lon'][np.argmin(abs(data.coords['lon']-longitude))])
                    elif longitude[1]>longitude[0]:
                        mask = mask&((data.coords['lon']>=longitude[0])&(data.coords['lon']<=longitude[1]))
                    else:
                        mask = mask&((data.coords['lon']>=longitude[0])|(data.coords['lon']<=longitude[1]))
                elif (isinstance(longitude,float) or isinstance(longitude,int)):
                    longitude=np.array(longitude)
                    if np.any([longitude<0,longitude>360]):
                        raise ValueError('"longitude" must be in the range [0÷360]')
                    else:
                        mask = mask&(data.coords['lon']==data.coords['lon'][np.argmin(abs(data.coords['lon']-longitude))])
                else:
                    raise Exception(lon_exc)
            else:
                mask=mask&(GREB.def_DataArray(np.full(GREB.dx(),True),dims='lon'))

        # Change values
        if (isinstance(value,float) or isinstance(value,int) or isinstance(value,np.ndarray)):
            data=data.where(~mask,value)
        elif isinstance(value,xr.DataArray):
            data=data.where(~mask,value.values)
        elif callable(value):
            data=data.where(~mask,value(data))
        else:
            raise Exception('"value" must be a number, numpy.ndarray, xarray.DataArray or function to apply to the "solar_base" (e.g. "lambda x: x*1.1")')
        # Correct value below 0
        data=data.where(data>=0,0)
        # Write .bin and .ctl files
        vars = {'solar':data.values}
        if outpath is None:
            outpath=GREB.solar_folder()+'/sw.artificial.ctl'
        GREB.create_bin_ctl(outpath,vars,tridimensional=tridimensional)

class Colormaps:
    def __init__(self):
        pass
    
    def add_white_inbetween(colormap,name=None):
        if name is None:
            name = 'new_'+colormap.name
        cm0=colormap
        col1=cm0(np.linspace(0,0.4,100))
        w1=np.array(list(map(lambda x: np.linspace(x,1,28),col1[-1]))).transpose()
        col2=cm0(np.linspace(0.6,1,100))
        w2=np.array(list(map(lambda x: np.linspace(1,x,28),col2[0]))).transpose()
        cols1=np.vstack([col1,w1,w2,col2])
        return colors.LinearSegmentedColormap.from_list(name, cols1)

    def add_white_start(colormap,name=None):
        if name is None:
            name = 'new_'+colormap.name
        col=colormap(np.linspace(0,1,200))
        w=np.array(list(map(lambda x: np.linspace(1,x,30),col[0]))).transpose()
        cols=np.vstack([w,col])
        return colors.LinearSegmentedColormap.from_list(name, cols)   

    def add_white_end(colormap,name=None):
        if name is None: name = 'new_'+colormap.name
        col=colormap(np.linspace(0,1,200))
        w=np.array(list(map(lambda x: np.linspace(x,1,30),col[-1]))).transpose()
        cols=np.vstack([col,w])
        return colors.LinearSegmentedColormap.from_list(name, cols) 

    def portion(colormap,range,name=None):
        if not isinstance(range,list) or isinstance(range,tuple):
            raise Exception("Range needs to be a list or tuple in the form [min, max].")
        if len(range) != 2:
            raise Exception("Range needs to be a list or tuple in the form [min, max].")
        if (range[0] < 0) or (range[1] > 1) or (range[0] >= range[1]): 
            raise Exception("Range must be an interval between [0 1].")
        if name is None: name = 'new_'+colormap.name
        n=int(np.round(colormap.N*(range[1]-range[0])))
        portion = colormap(np.linspace(*range, n))
        return colors.LinearSegmentedColormap.from_list(name, portion)

    div_tsurf = add_white_inbetween(cm.Spectral_r,name='div_tsurf')
    div_tsurf_r = div_tsurf.reversed()
    div_precip = add_white_inbetween(cm.twilight_shifted_r,name='div_precip')
    div_precip_r = div_precip.reversed()
    seq_tsurf_hot = portion(div_tsurf,[0.5,1], name="seq_tsurf_hot")
    seq_tsurf_hot_r = seq_tsurf_hot.reversed()
    seq_tsurf_cold = portion(div_tsurf,[0,0.5], name="seq_tsurf_cold")
    seq_tsurf_cold_r = seq_tsurf_cold.reversed()
    seq_precip_wet = portion(div_precip,[0.5,1], name="seq_precip_wet")
    seq_precip_wet_r = seq_precip_wet.reversed()
    seq_precip_dry = portion(div_precip,[0,0.5], name="seq_precip_dry")
    seq_precip_dry_r = seq_precip_dry.reversed()

class SREX_regions:
    def __init__(self):
        pass
    
    abbrevs = srex_regions.abbrevs
    names = srex_regions.names

    @staticmethod
    def mask(name=None):
        if name is None: name='srex_region'
        return DataArray(srex_regions.mask(GREB.def_DataArray(dims=['lat','lon']),wrap_lon=True),name=name)

    @staticmethod
    def plot(text_kws = dict(), **kwargs):
        if 'label' not in kwargs: kwargs['label']='abbrev'
        if 'add_ocean' not in kwargs: kwargs["add_ocean"]=True
        if "fontsize" in kwargs: text_kws["fontsize"] = kwargs.pop('fontsize')
        if "fontsize" not in text_kws:
            text_kws["fontsize"] = 8
        srex_regions.plot(text_kws=text_kws,**kwargs)

def group_by(x,time_group,copy=True,update_attrs=True):
    """
    Group an xarray.Dataset object according to a specific time group and compute
    the mean over the values.

    Arguments
    ----------
    x : xarray.DataArray or xarray.Dataset
        Array or Dataset to group
    time_group : str
        Time grouping method to be chosen between: '12h','day','month','year','season'.
        If chosen, the retrieved data belonging to the same time_group will be
        averaged.
        If "time_group" is smaller than the data time-resolution, a spline
        interpolation is performed.

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that it was parsed with the "parse_greb_var" function.

    Returns
    ----------
    xarray.DataArray or xarray.Dataset
         Grouped Array or Dataset.

    """

    if copy: x = x.copy()
    attrs = x.attrs
    if 'grouped_by' in attrs:
        raise Exception('Cannot group an Array which has already been grouped.')
    if update_attrs:
        attrs['grouped_by'] = time_group
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['grouped_by'] = time_group
    interp=False
    if time_group == '12h':
        nt=730
        time_group = 'month'
        interp=True
    elif time_group == 'day':
        nt=365
        time_group = 'month'
        interp=True
    x = x.groupby('time.{}'.format(time_group)).mean(dim='time',keep_attrs=True).rename({'{}'.format(time_group):'time'})
    if interp:
        x = x.interp(time=np.linspace(x.time[0]-0.5,x.time[-1]+0.5,nt),method='cubic',kwargs={'fill_value':'extrapolate'}).assign_coords(time=GREB.t()[::int(730/nt)])
    f = DataArray if check_xarray(x,'DataArray') else Dataset
    return f(x,attrs=attrs)

def rmext(filename):
    """
    Remove the .bin,.gad or .ctl extension at the end of the filename.
    If none of those extensions is present, returns filename.

    Arguments
    ----------
    filename : str
        Path to remove the extension from

    Returns
    ----------
    str
        New path with ".bin" or ".ctl" extension removed.

    """

    path,ext = os.path.splitext(filename)
    if ext not in ['.ctl','.bin','.gad','.']: path = path+ext
    return path

def bin2netCDF(file):
    """
    Convert a binary (".bin") file to a netCDF (".nc") file.

    Arguments
    ----------
    file : str
        Path of the file (with or without ".bin") to convert to netCDF.

    Returns
    ----------
    None
        Creates a new netCDF file with the same name as the original binary

    """

    filename = rmext(file)
    cdo = Cdo() # Initialize CDO
    cdo.import_binary(input = filename+'.ctl', output = filename+'.nc',
                      options = '-f nc')

def exception_xarray(type = None,varname = 'x'):
    '''
    Prescribed xarray exception.

    Parameters
    ----------
    type : str
       type of xarray object: 'DataArray' or 'Dataset'
    varname : str
        name of the variable to display in the exception

    Returns
    ----------
    ValueError Exception

    '''

    if type is not None:
        if type.lower() == 'dataarray':
            raise ValueError('{} must be an xarray.DataArray object'.format(varname))
        elif type.lower() == 'dataset':
            raise ValueError('{} must be an xarray.Dataset object'.format(varname))
    raise ValueError('{} must be either an xarray.DataArray or xarray.Dataset object'.format(varname))

def check_xarray(x,type=None):
    '''
    Check if the argument is an xarray object.

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to be checked

    Parameters
    ----------
    type : str
       type of xarray object: 'DataArray' or 'Dataset'
    varname : str
       name of the variable to display in the exception

    Returns
    ----------
    Bool

    '''

    da = isinstance(x,xr.DataArray)
    ds = isinstance(x,xr.Dataset)
    if type is not None:
        if type.lower() == 'dataarray': return da
        elif type.lower() == 'dataset': return ds
    return np.logical_or(da,ds)

def parse_greb_var(x,update_attrs=True):
    '''
    Corrects GREB model output variables and adds units label

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array of GREB output variables to be parsed

    Parameters
    ----------
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that it was parsed with the "parse_greb_var" function.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray depending on the argument's type

    '''

    def _parsevar(x,update_attrs=None):
        name = x.name
        # TATMOS,TSURF,TOCEAN
        if name == 'tatmos':
            x.attrs['long_name']='Atmospheric Temperature'
            x.attrs['units']='K'
        elif name == 'tsurf':
            x.attrs['long_name']='Surface Temperature'
            x.attrs['units']='K'
        elif name == 'tocean':
            x.attrs['long_name']='Ocean Temperature'
            x.attrs['units']='K'
        elif name == 'precip':
            x.attrs['long_name']='Precipitation'
            x.attrs['units']='mm/day'
            x*=-86400
        elif name == 'eva':
            x.attrs['long_name']='Evaporation'
            x.attrs['units']='mm/day'
            x*=-86400
        elif name == 'qcrcl':
            x.attrs['long_name']='Circulation'
            x.attrs['units']='mm/day'
            x*=-86400
        elif name == 'vapor':
            x.attrs['long_name']='Specific Humidity'
            x.attrs['units']=''
        elif name == 'ice':
            x.attrs['long_name']='Ice'
            x.attrs['units']=''
        elif name == 'sw':
            x.attrs['long_name']='SW Radiation Output'
            x.attrs['units']='W/m2'
        elif name == 'cloud':
            x.attrs['long_name']='Clouds'
            x.attrs['units']=''
        elif name == 'solar':
            x.attrs['long_name']='SW Radiation Input'
            x.attrs['units']='W/m2'

        if update_attrs:
            x.attrs['parse_greb_var']='Parsed with parse_greb_var function'
        return x

    if 'parse_greb_var' in x.attrs: return x
    if check_xarray(x,'dataarray'):
        return DataArray(_parsevar(x,update_attrs).squeeze())
    elif check_xarray(x,'dataset'):
        x.apply(lambda a: _parsevar(a,update_attrs),keep_attrs=True)
        if update_attrs:
            x.attrs['parse_greb_var']='Parsed with parse_greb_var function'
        return Dataset(x,attrs=x.attrs).squeeze()
    else: exception_xarray()

def average(x, dim=None, weights=None,**kwargs):
    """
    weighted average for DataArrays

    Arguments
    ----------
    x : xarray.DataArray
        array to compute the average on

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of x.

    Returns
    ----------
    reduced : DataArray
        New DataArray with average applied to its data and the indicated
        dimension(s) removed.

    """

    if not check_xarray(x,'DataArray'): exception_xarray('DataArray')
    attrs=x.attrs
    if weights is None:
        return x.mean(dim,**kwargs)
    else:
        if not check_xarray(weights,'DataArray'):
            exception_xarray(type='DataArray',varname='weights')

        # if NaNs are present, we need individual weights
        if not x.notnull().all():
            total_weights = weights.where(x.notnull()).sum(dim=dim,**kwargs)
        else:
            total_weights = weights.sum(dim,**kwargs)
        numerator = xr.apply_ufunc(lambda a,b: a*b,x, weights,**kwargs).sum(dim,**kwargs)
        return DataArray(xr.apply_ufunc(lambda a,b: a/b,numerator, total_weights,**kwargs),attrs=attrs)

def rms(x,copy=True,update_attrs=True):
    '''
    Compute the root mean square error over "lat" and "lon" dimension.

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the rms on

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "rms" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with rms applied to its "lat" and "lon"
    dimension.

    '''

    if not check_xarray(x): exception_xarray()
    if 'rms' in x.attrs:return x
    if 'global_mean' in x.attrs:
        raise Exception('Cannot perform rms on a variable on which global mean has already been performed')
    if copy: x = x.copy()
    func=DataArray
    if check_xarray(x,'Dataset'):
        if update_attrs:
            for var in x: x._variables[var].attrs['rms'] = 'Computed root mean square'
        func=Dataset
    attrs=x.attrs
    gm=global_mean(x**2,update_attrs=False)
    if update_attrs:
        attrs['rms'] = 'Computed root mean square'
    return func(xr.apply_ufunc(lambda x: np.sqrt(x),gm,keep_attrs=True),attrs=attrs)

def annual_mean(x,num=None,copy=True,update_attrs=True):
    '''
    Compute the mean over 'time' dimension.

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the annual mean on

    Parameters
    ----------
    num : int
        If set to None (default), compute the mean over all the time coordinate.
        Otherwise it takes into account only the last "num_year" timesteps.
        If num years > len(time coordinate), the mean gets computed over all the time coordinate.
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the annual_mean function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with average applied to its "time" dimension.

    '''

    if not check_xarray(x): exception_xarray()
    if 'annual_mean' in x.attrs: return x
    if 'seasonal_cycle' in x.attrs:
        raise Exception('Cannot perform annual mean on a variable on which seasonal cycle has already been performed')
    if copy: x = x.copy()
    if num is None: 
        num = len(x['time'])
    elif num > len(x['time']): 
        num = len(x['time'])
    if update_attrs:
        x.attrs['annual_mean'] = 'Computed annual mean of {} timesteps'.format(num)
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['annual_mean'] = 'Computed annual mean of {} timesteps'.format(num)
    x = x.isel(time=slice(-num,None))
    return x.mean(dim='time',keep_attrs=True).squeeze()

def annual_cycle(x,num=None,copy=True,update_attrs=True):
    '''
    Compute the annual cycle over 'time' dimension.

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the annual mean on

    Parameters
    ----------
    num : int
        If set to None (default), compute the annual cycle over all the time coordinate.
        Otherwise it takes into account only the last "num_year" timesteps.
        If num years > len(time coordinate), the annual cycle gets computed over all the time coordinate.
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the annual cycle function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with annual cycle applied to its "time" dimension.

    '''

    if not check_xarray(x): exception_xarray()
    if 'annual_cycle' in x.attrs: return x
    if 'seasonal_cycle' in x.attrs:
        raise Exception('Cannot perform annual cycle on a variable on which seasonal cycle has already been performed')
    if 'annual_mean' in x.attrs:
        raise Exception('Cannot perform annual cycle on a variable on which annual mean has already been performed')
    if copy: x = x.copy()
    if num is None: 
        num = len(x['time'])
    elif num > len(x['time']): 
        num = len(x['time'])
    if update_attrs:
        x.attrs['annual_cycle'] = 'Computed annual cycle of {} timesteps'.format(num)
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['annual_cycle'] = 'Computed annual cycle of {} timesteps'.format(num)
    x = x.isel(time=slice(-num,None))
    return x.groupby("time.dayofyear").mean("time",keep_attrs=True).rename({"dayofyear":"time"}).squeeze()

def latitude_mean(x,copy=True,update_attrs=True):
    '''
    Compute the mean over latitude dimension, weigthed with cos(latitude).

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the global mean on

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "global_mean" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    weights.
    New Dataset or DataArray object with average applied to its latitude, weighted with cos(latitude).

    '''
    if not check_xarray(x): exception_xarray()
    lat,lon=x.get_spatial_coords()
    if 'latitude_mean' in x.attrs: return x
    if 'global_mean' in x.attrs:
        raise Exception('Cannot perform the mean over latitude on a variable on which global mean has already been performed')
    if copy: x = x.copy()
    if update_attrs:
        x.attrs['latitude_mean'] = 'Computed latitude mean'
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['latitude_mean'] = 'Computed latitude mean'
    if lat in x.dims:
        weights = np.cos(np.deg2rad(x[lat]))
        return x.average(dim=lat,weights=weights,keep_attrs=True).squeeze()
    else:
        raise Exception('Impossible to perform latitude mean, no latitude dim.')

def global_mean(x,copy=True,update_attrs=True):
    '''
    Compute the global mean over 'lat' and 'lon' dimension.
    The average over 'lat' dimension will be weigthed with cos(lat).

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the global mean on

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "global_mean" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with average applied to its "lat" and
    "lon" dimension. The average along "lat" dimesnion is weighted with cos(lat)
    weights.

    '''
    if not check_xarray(x): exception_xarray()
    lat,lon=x.get_spatial_coords()
    if 'global_mean' in x.attrs: return x
    lat_mean = True if 'latitude_mean' in x.attrs else False
    if 'rms' in x.attrs:
        raise Exception('Cannot perform global mean on a variable on which rms has already been performed')
    if copy: x = x.copy()
    if update_attrs:
        x.attrs['global_mean'] = 'Computed global mean'
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['global_mean'] = 'Computed global mean'
            if lat_mean: 
                for var in x: del x._variables[var].attrs['latitude_mean']
        if lat_mean: del x.attrs["latitude_mean"]
    if lat in x.dims and lon in x.dims:
        if not lat_mean:
            weights = np.cos(np.deg2rad(x[lat]))
            return x.average(dim=lat,weights=weights,keep_attrs=True).mean(lon,keep_attrs=True).squeeze()
        else:
            return x.mean(lon,keep_attrs=True).squeeze()
    elif 'stacked_lat_lon' in x.coords:
        weights = np.cos(np.deg2rad(x[lat]))
        return x.average(dim='stacked_lat_lon',weights=weights,keep_attrs=True).squeeze()
    else:
        raise Exception('Impossible to perform global mean, no latitude and longitude dims.')

def seasonal_cycle(x,copy=True,update_attrs=True):
    '''
    Compute the seasonal cycle (DJF-JJA) over time dimension

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the seasonal cycle on

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "seasonal_cycle" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with seasonal cycle applied to its "time" dimension.

    '''

    if not check_xarray(x): exception_xarray()
    if 'seasonal_cycle' in x.attrs: return x
    if 'annual_mean' in x.attrs:
        raise Exception('Cannot perform seasonal cycle on a variable on which annual mean has already been performed')
    if 'global_mean' in x.attrs:
        raise Exception('Cannot perform annual mean on a variable on which global mean has already been performed')
    if 'rms' in x.attrs:
        raise Exception('Cannot perform seasonal cycle on a variable on which rms has already been performed')
    if copy: x = x.copy()
    f=DataArray
    if update_attrs:
        x.attrs['seasonal_cycle'] = 'Computed seasonal cycle'
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['seasonal_cycle'] = 'Computed seasonal cycle'
            f=Dataset
    attrs=x.attrs
    x_seas=f(x.group_by('season'),attrs=attrs)
    x_seas=(x_seas.sel(time='DJF')-x_seas.sel(time='JJA'))/2
    return x_seas.drop('time') if 'time' in x_seas.coords else x_seas

def anomalies(x,x_base=None,copy=True,update_attrs=True):
    '''
    Compute anomalies of x with respect to x_base (x-x_base).

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the anomalies on
    x_base : xarray.Dataset or xarray.DataArray object
       array to compute the anomalies from.
       x and x_base variables and dimensions must match.

    Parameters
    ----------
    copy : Bool
       set to True (default) if you want to return a copy of the argument in
       input; set to False if you want to overwrite the input argument.
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "anomalies" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object being the difference between x and x_base

    '''

    def fun(y):
        attrs = y.attrs
        dims = y.dims
        coords = dict(y.coords)
        coords['time'] = x.coords['time'].values
        return xr.DataArray(np.tile(y,(int(x.time.shape[0]/12),1,1)),
                     coords=coords, dims=dims, attrs=attrs)

    if not check_xarray(x): exception_xarray()
    if x_base is None:
        if (check_xarray(x,'DataArray') and x.name == 'cloud') or \
            (check_xarray(x,'Dataset') and 'cloud' in list(x.data_vars.keys())):
            ctrfile=GREB.cloud_def_file()
        elif (check_xarray(x,'DataArray') and x.name == 'solar') or \
            (check_xarray(x,'Dataset') and 'solar' in list(x.data_vars.keys())):
            try: len(x.lon)
            except TypeError: ctrfile=GREB.solar_def_file(tridimensional=False)
            else: ctrfile=GREB.solar_def_file(tridimensional=True)
        else:
            ctrfile=GREB.control_def_file()
        if 'grouped_by' in x.attrs:
            x_base = GREB.from_binary(ctrfile,x.attrs['grouped_by'])
        elif (any(['annual_mean' in x.attrs,'seasonal_cycle' in x.attrs,'global_mean' in x.attrs,'rms' in x.attrs,'latitude_mean'])) or \
            (ctrfile == GREB.cloud_def_file()) or (ctrfile == GREB.solar_def_file()):
            x_base = GREB.from_binary(ctrfile)
        else:
            x_base = GREB.from_binary(ctrfile).apply(fun,keep_attrs=True)
        # Change to Celsius if needed
        temp = ['tsurf','tocean','tatmos']
        if check_xarray(x,'DataArray'):
            if x.name in temp:
                if x.attrs['units'] == 'C': x_base = x_base.to_celsius(copy=False)
        else:
            vars=[d for d in x]
            for t in temp:
                if (t in vars) and (x[t].attrs['units'] == 'C'):
                    x_base = x_base.to_celsius(copy=False)
    else:
        if not check_xarray(x_base): exception_xarray()
    if 'annual_mean' in x.attrs: 
        num= int(x.attrs['annual_mean'].split()[4])
        x_base = annual_mean(x_base,num=num)
    if 'seasonal_cycle' in x.attrs: x_base = seasonal_cycle(x_base)
    if 'global_mean' in x.attrs: x_base = global_mean(x_base)
    if 'rms' in x.attrs: x_base = rms(x_base)
    if copy: x = x.copy()
    if update_attrs:
        x.attrs['anomalies'] = 'Anomalies'
        if check_xarray(x,'Dataset'):
            for var in x: x._variables[var].attrs['anomalies'] = 'Anomalies'
    return x-x_base

def to_Robinson_cartesian(lat,lon,lon_center = 0):
    '''
    Convert lat/lon points to Robinson Projection.

    Arguments
    ----------
    lat : float or array of floats
        latitude point or array of latitude points.
    lon : float or array of floats
        longitude point or array of longitude points.
        lat and lon must be the same lenght

    Parameters
    ----------
    lon_center : float
        center meridiane of the Robinson projection

    Returns
    ----------
    float or array of floats
        Robinson projection projected points

    '''

    from scipy.interpolate import interp1d
    center = np.deg2rad(lon_center)
    lat=np.array(lat)
    if np.any(lat<-90) or np.any(lat>90):
        raise ValueError('lat value must be between -90 and 90 degrees')
    lon=np.array(lon)
    if np.any(lon<0) or np.any(lon>360):
        raise ValueError('lon value must be between 0 and 360 degrees')
    lon = np.deg2rad(lon)
    R = 6378137.1
    lat_def = np.arange(0,95,5)
    X_def= [1,0.9986,0.9954,0.9900,0.9822,0.9730,0.9600,0.9427,0.9216,0.8962,
        0.8679,0.8350,0.7986,0.7597,0.7186,0.6732,0.6213,0.5722,0.5322]
    Y_def= [0,0.0620,0.1240,0.1860,0.2480,0.3100,0.3720,0.4340,0.4958,0.5571,
        0.6176,0.6769,0.7346,0.7903,0.8435,0.8936,0.9394,0.9761,1.0000]

    iX = interp1d(lat_def, X_def, kind='cubic')
    iY = interp1d(lat_def, Y_def, kind='cubic')
    x = 0.8487*R*iX(np.abs(lat))*(lon-lon_center)
    y = 1.3523*R*iY(np.abs(lat))
    x=np.where((lon % 360) > 180,-x,x)
    y=np.where(lat < 0,-y,y)

    return x,y

def _check_shapes(x1,x2):
    '''
    Check units and coords matching between Datarrays or Datasets
    '''
    # CHECK COORDS
    if not (x1.shape == x2.shape):
        raise Exception("Shapes don't match!")

def seasonal_time_series(x,first_month_num=None,update_attrs=True):
    '''
    Compute the seasonal average and return the seasonal time series.

    Arguments
    ----------
    x : xarray.Dataset or xarray.DataArray object
       array to compute the seasonal average on

    Parameters
    ----------
    first_month_num : Int
        Number of month in the year for first x value
        (1 = Jan, 2 = Feb ... 11 = Nov, 12 = Dec)
    update_attrs : Bool
        If set to True (default), the new DataArray/Dataset will have an attribute
        as a reference that the "seasonal_time_series" function has been applied to it.

    Returns
    ----------
    xarray.Dataset or xarray.DataArray

    New Dataset or DataArray object with average applied to its "time" dimension, every 3 values.
    '''
    def check_first_months(x,first_month_num):
        length=len(x.coords['time'])
        # if first month is Dec,Mar,Jun or Sep, start is None
        if first_month_num in (12,3,6,9):
            start = None
        # if first month is Oct,Jan,Apr or Jul, start = average of first 2 values
        elif first_month_num in (10,1,4,7):
            start = x.isel(time=[0,1]).mean('time',keep_attrs=True).expand_dims('time',axis=0)
            x = x.isel(time=slice(2,None))
            # if first month is Nov,Feb,May or Aug, start = first value
        elif first_month_num in (11,2,5,8):
            start = x.isel(time=0).expand_dims('time',axis=0).drop('time')
            x = x.isel(time=slice(1,None))
        return [start,x]

    def check_last_months(x):
        length=len(x.coords['time'])
        last_months=length % 3
        # if last month is Nov,Feb,May or Aug, end is None
        if last_months == 0:
            end = None
        # if last month is Dec,Mar,Jun or Sep, end = first value
        elif last_months == 1:
            end = x.isel(time=-1).expand_dims('time',axis=0).drop('time')
            x = x.isel(time=slice(None,-1))
        # if last month is Oct,Jan,Apr or Jul, end = average of last 2 values
        elif last_months == 2:
            end = x.isel(time=[-2,-1]).mean('time',keep_attrs=True).expand_dims('time',axis=0)
            x = x.isel(time=slice(None,-2))
        return [end,x]

    def mean_each_N(x,N=3):
        length=len(x.coords['time'])
        ind=[0]+list(np.repeat(np.arange(N,length,N),2))+[length-1]
        newdata=xr.apply_ufunc(lambda x: np.add.reduceat(x,ind,axis=-1),
             x,
             input_core_dims=[['time']],
             output_core_dims=[['time']],
             exclude_dims={'time'},
             keep_attrs=True,
             ).isel(time=slice(None,None,2))
        return newdata

    if not check_xarray(x): exception_xarray()
    dims=[d for d in x.dims]
    # compute x's first month (if 'first_month_num' not explicitly given as optional input)
    if first_month_num is None:
        first_month_num=dtime.utcfromtimestamp((x.time[0].values-np.datetime64('1970-01-01T00:00:00Z'))/np.timedelta64(1, 's')).month
    start,x=check_first_months(x,first_month_num)
    end,x=check_last_months(x)
    middle=mean_each_N(x)
    newdata=xr.concat([a for a in [start,middle,end] if a is not None],dim='time',coords='all')
    if update_attrs:
        newdata.attrs['seasonal_time_series'] = 'Computed seasonal time series'
        if check_xarray(newdata,'Dataset'):
            for var in x: x._variables[var].attrs['seasonal_time_series'] = 'Computed seasonal time series'
    dims.remove('time')
    return newdata.transpose('time',*dims)

def t_student_probability(x,y,season=None):

    '''
    Perform the t-student test over 2 different Datarrays (groups of samples).
    Among each group, every simulation along the specified dimension, is considered
    to be an independent sample.
    Returns the probability value associated with the t-distribution.

    Arguments
    ----------
    x,y : DataArray objects
       arrays to compute the t-student test on

    Parameters
    ----------
    season : ['DJF','MAM','JJA','SON']
    Season to group data before computing the probabilities.

    Returns
    ----------
    xarray.DataArray

    New DataArray containing the probabilities associated with the t-distribution for
    the two Dataarrays, along the specified dimension.
    '''
    if season is not None:
        if season in ['DJF','MAM','JJA','SON']:
            x1 = x.isel(time=x.groupby('time.season').groups[season])
            y1 = y.isel(time=y.groupby('time.season').groups[season])
        else:
            raise Exception("season must be a value among 'DJF','MAM','JJA' and 'SON'.")
    try:
        x1=x.groupby('time.year').mean('time')
        y1=y.groupby('time.year').mean('time')
    except:
        pass
    
    p=stats.ttest_ind(x1,y1,nan_policy="omit").pvalue
    p[np.where(np.isnan(p))]=1
    dims=list(x.dims)
    dims.remove('time')
    coords=[x[a].values for a in dims]
    p=DataArray(data=p,dims=dims,coords=coords)
    return p

def open_dataarray(x,**open_dataarray_kwargs):
    '''
    Function analogous to xarray.open_dataarray which wraps the result in the custom DataArray class 
    '''

    return DataArray(xr.open_dataarray(x,**open_dataarray_kwargs))

def open_mfdataset(x,**open_mfdataset_kwargs):
    '''
    Function analogous to xarray.open_mfdataset which parse the input, applying the following functions:
        - Add_evaporation
        - Rename_m01s09i231
        - Split_w_wind
    '''
    d=xr.open_mfdataset(x,**open_mfdataset_kwargs)
    if ("evaporation_flux_from_open_sea" in d) and ("evaporation_from_soil_surface" in d):
        d=UM.add_evaporation(d)
    if "m01s09i231" in d:
        d=UM.rename_m01s09i231(d)
    if "upward_air_velocity" in d:
        d=UM.split_w_wind(d)
    return d
# ============================================================================ #
# ============================================================================ #
# ============================================================================ #
# ============================================================================ #
